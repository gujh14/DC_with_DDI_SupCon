import pickle
from MSI.load_msi_data import LoadData

import torch
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path

from rdkit import Chem
from mordred import Calculator, descriptors

from torch.utils.data import Dataset, DataLoader

class CombinationDatasetRW(Dataset):
    '''
    Drug combination dataset for random walk algorithms
    '''
    def __init__(self, database='DC_combined', kgfeat=None, chemfeat=None, neg_ratio=1, neg_dataset='random', seed=42, transform=None, exclude_list=[]):
        '''
        args
            - database: str, default='DC_combined' ['DC_combined', 'DC_combined_small']
            - kgfeat: str, default=None [None, 'node2vec', 'edge2vec', 'res2vec_homo', 'res2vec_hetero', 'DREAMwalk', 'NEWMIN']
            - chemfeat: str, default=None [None, 'ecfp', 'maccs', 'mordred']
            - neg_ratio: int, default=1
            - neg_dataset: str, default='random' ['random', 'TWOSIDES']
            - seed: int, default=42
            - exclude_list: list, default=[] (list of DB ID sets to exclude from the training set)
        '''
        if database not in ['DC_combined', 'DC_combined_small']:
            raise ValueError('database must be one of [DC_combined, DC_combined_small]')
        if neg_ratio < 1:
            raise ValueError('neg_ratio must be greater than 1')
        if neg_dataset not in ['random', 'TWOSIDES']:
            raise ValueError('neg_dataset must be one of [random, TWOSIDES]')
        if (chemfeat is not None) and (database != 'DC_combined_small'):
            raise ValueError('chemical feature is only fully supported for DC_combined_small dataset.')
        
        self.database = database
        self.kgfeat = kgfeat
        self.chemfeat = chemfeat
        self.neg_ratio = neg_ratio
        self.transform = transform
        self.neg_dataset = neg_dataset
        self.seed = seed
        self.exclude_list = exclude_list
        
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
            
        if len(exclude_list) == 0:
            self.data_path = Path('data/processed')/f'{database}_kgfeat({kgfeat})_chemfeat({chemfeat})_neg({neg_dataset}_{neg_ratio})_seed{seed}.pt'
        else:
            self.data_path = Path('data/processed')/f'casestudy_{database}_kgfeat({kgfeat})_chemfeat({chemfeat})_neg({neg_dataset}_{neg_ratio})_seed{seed}.pt'

        if self.data_path.exists():
            print(f'{self.data_path} already exists in processed/ directory.')
        else:
            self._process()
        
        print(f'Loading dataset...{self.data_path}')
        print('Dictionary of {train, valid, test, whole} dataset is loaded.')
        self.data = torch.load(self.data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _process(self):
        print('Processing dataset...')
        dataset_list = self._create_dataset()
        train_size = int(len(dataset_list) * 0.8)
        valid_size = int(len(dataset_list) * 0.1)
        test_size = len(dataset_list) - train_size - valid_size
        torch.manual_seed(0)
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset_list, [train_size, valid_size, test_size])
        dataset_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset, 'whole': dataset_list}

        print(f'Saving dataset...{self.data_path}')
        torch.save(dataset_dict, self.data_path)

    def _create_dataset(self):
        dataloader = LoadData()
        # Knowledge graph embedding
        drug_features = []
        if self.kgfeat is not None:
            with open(f'data/embedding/embeddings_{self.kgfeat}_msi_seed0.pkl', 'rb') as f:
                kgfeat_dict = pickle.load(f)
            drug_features.append(kgfeat_dict)
        # Chemical feature embedding
        if self.chemfeat is not None:
            # chemical feature embedding
            with open(f'data/embedding/{self.chemfeat}_dict.pickle', 'rb') as f:
                chemfeat_dict = pickle.load(f)
            drug_features.append(chemfeat_dict)
        
        if self.database == 'DC_combined':
            pos_df = pd.read_csv('data/labels/DC_combined_msi.tsv', sep='\t')
        elif self.database == 'DC_combined_small':
            pos_df = pd.read_csv(f'data/labels/DC_combined_msi_small.tsv', sep='\t') # biologics drugs are excluded (no SMILES, no chemical features)
        
        # Drug dictionary
        drug_id2name, drug_name2id = dataloader.get_dict(type='drug')

        dataset_list = []
        # Prepare positive labels
        for i in range(len(pos_df)):
            drug1_id = pos_df.iloc[i, 0]
            drug2_id = pos_df.iloc[i, 1]
            if set([drug1_id, drug2_id]) in self.exclude_list:
                continue
            drug1_embedding = np.concatenate([feat[drug1_id] for feat in drug_features])
            drug2_embedding = np.concatenate([feat[drug2_id] for feat in drug_features])
            comb_embedding = np.concatenate([drug1_embedding, drug2_embedding])
            dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(1, dtype=torch.long)])
        
        # Prepare negative labels
        if self.neg_dataset == 'random':
            count = 0
            while count < len(pos_df) * self.neg_ratio:
                drug1_id = random.choice(list(drug_id2name.keys()))
                drug2_id = random.choice(list(drug_id2name.keys()))
                if set([drug1_id, drug2_id]) in self.exclude_list:
                    continue
                if drug1_id == drug2_id:
                    continue
                if ((pos_df['drug_1'] == drug1_id) & (pos_df['drug_2'] == drug2_id)).any():
                    continue
                if ((pos_df['drug_1'] == drug2_id) & (pos_df['drug_2'] == drug1_id)).any():
                    continue
                if self.chemfeat is not None:
                    db_lst = list(chemfeat_dict.keys())
                    if drug1_id not in db_lst or drug2_id not in db_lst:
                        continue
                drug1_embedding = np.concatenate([feat[drug1_id] for feat in drug_features])
                drug2_embedding = np.concatenate([feat[drug2_id] for feat in drug_features])
                comb_embedding = np.concatenate([drug1_embedding, drug2_embedding])
                dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
                count += 1
        elif self.neg_dataset == 'TWOSIDES':
            neg_df = pd.read_csv(f'data/labels/TWOSIDES_msi.tsv', sep='\t')
            if len(neg_df) < len(pos_df) * self.neg_ratio:
                raise ValueError('Not enough negative samples in TWOSIDES dataset')
            
            neg_df = neg_df.sample(n=len(pos_df) * self.neg_ratio, random_state=self.seed)
            for i in range(len(neg_df)):
                drug1_id = neg_df.iloc[i, 0]
                drug2_id = neg_df.iloc[i, 1]
                if set([drug1_id, drug2_id]) in self.exclude_list:
                    continue
                drug1_embedding = np.concatenate([feat[drug1_id] for feat in drug_features])
                drug2_embedding = np.concatenate([feat[drug2_id] for feat in drug_features])
                comb_embedding = np.concatenate([drug1_embedding, drug2_embedding])
                dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
        
        return dataset_list