import torch
import numpy as np
import os
import random

import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import from_networkx

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from model import CombGNN

import networkx as nx

import pickle
import argparse
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        '''
        args:
            - patience: int, default=10
            - verbose: bool, default=False
            - delta: float, default=0
            - path: str, default='checkpoint.pt'
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def read_graph(edgelist, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgelist, nodetype=str, data=(('type',int),('weight',float),('id',int)), create_using=nx.MultiDiGraph())
    else:
        G = nx.read_edgelist(edgelist, nodetype=str,data=(('type',int),('id',int)), create_using=nx.MultiDiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1.0

    if not directed:
        G = G.to_undirected()
    
    return G

def parse_args():
    parser = argparse.ArgumentParser(description='CombNet_GNN')
    parser.add_argument('--database', type=str, default='DC_combined', choices=['C_DCDB', 'DCDB', 'DC_combined'])
    parser.add_argument('--conv', type=str, default='GCN', choices=['GCN', 'SAGE', 'GAT', 'GIN'])
    parser.add_argument('--neg_dataset', type=str, default='TWOSIDES', choices=['TWOSIDES', 'random'])
    parser.add_argument('--comb_type', type=str, default='prod_fc', choices=['sum', 'cosine', 'prod_fc'])
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--ce_lr', type=float, default=1e-3)
    parser.add_argument('--contra_lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--neg_ratio', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_mode', type=str, default='contra', choices=['contra', 'nocontra'])
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False, help='use wandb or not')
    parser.add_argument('--entity', type=str, help='your wandb entity name')
    parser.add_argument('--project', type=str, help='your wandb project name')

    args = parser.parse_args()
    return args

def train_contrastive(model, device, pyg_graph, edge_label_index, edge_label, optimizer):
    '''
    Contrastive training with info nce loss
    '''
    model.train()

    optimizer.zero_grad()

    pyg_graph = pyg_graph.to(device)
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)

    cos_sim = model.forward_contrastive(pyg_graph.x, pyg_graph.edge_index, edge_label_index)
    info_nce_loss = model.info_nce_loss(cos_sim, edge_label)
    info_nce_loss.backward()
    optimizer.step()
    train_loss = info_nce_loss.item()

    return train_loss

def train_ce(model, device, pyg_graph, edge_label_index, edge_label, criterion, optimizer, metric_list=[accuracy_score]):
    '''
    Cross entropy training
    '''
    model.train()
    optimizer.zero_grad()

    pyg_graph = pyg_graph.to(device)
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)

    out = model(pyg_graph.x, pyg_graph.edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    target_list = edge_label.long().detach().cpu().numpy()
    pred_list = torch.sigmoid(out).detach().cpu().numpy()

    scores = []
    for metric in metric_list:
        if (metric == roc_auc_score) or (metric == average_precision_score):
            scores.append(metric(target_list, pred_list))
        else:
            scores.append(metric(target_list, pred_list.round()))
    
    return train_loss, scores

def evaluate(model, device, pyg_graph, edge_label_index, edge_label, criterion, metric_list=[accuracy_score], checkpoint=None):
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.eval()
    
    pyg_graph = pyg_graph.to(device)
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)
    with torch.no_grad():
        out = model(pyg_graph.x, pyg_graph.edge_index, edge_label_index).view(-1)
        eval_loss = criterion(out, edge_label).item()
        target_list = edge_label.long().detach().cpu().numpy()
        pred_list = torch.sigmoid(out).detach().cpu().numpy()

        scores = []
        for metric in metric_list:
            if (metric == roc_auc_score) or (metric == average_precision_score):
                scores.append(metric(target_list, pred_list))
            else:
                scores.append(metric(target_list, pred_list.round()))
        
    return eval_loss, scores

def create_pyg_graph(graph_name):
    graph_dir = 'MSI/network_files/msi_network.txt'
    print('Reading graph...')
    G = read_graph(graph_dir, weighted=True, directed=False)

    pyg_graph = from_networkx(G)
    pyg_graph.x = torch.LongTensor([i for i in range(pyg_graph.num_nodes)])
    pyg_graph.nodes = list(G.nodes())
    torch.save(pyg_graph, graph_name)

def main():
    args = parse_args()
    if args.wandb:
        group = f'{args.train_mode}_{args.conv}{args.nlayers}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})_celr({args.ce_lr})_contralr({args.contra_lr})'
        wandb.init(project=args.project, group=group, entity=args.entity)
        wandb.config.update(args)
        wandb.run.name = f'{args.train_mode}_{args.conv}{args.nlayers}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})_seed{args.seed}'
        wandb.run.save()
    print(args)

    seed_everything(args.seed)

    # load graph (msi network)
    graph_name = f'data/processed/pyg_graph_msi.pt'
    if os.path.exists(graph_name):
        print(f'{graph_name} exists, loading data from file...')
    else:
        print(f'{graph_name} does not exist, creating graph file...')
        create_pyg_graph(graph_name)
    pyg_graph = torch.load(graph_name)
    nodes = pyg_graph.nodes

    # load dc, (ddi or random) pairs from split
    examples = {}
    y = {}
    modes = ['train', 'valid', 'test']
    with open(f'data/splits_gcn/DC_neg({args.neg_dataset}_{args.neg_ratio})_split{args.seed}.pkl', 'rb') as f:
        split_dict = pickle.load(f)
        pairs = split_dict['pairs']
        labels = split_dict['labels']
    
    examples['train'], examples['valid'], y['train'], y['valid'] = train_test_split(pairs, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    examples['test'], examples['valid'], y['test'], y['valid'] = train_test_split(examples['valid'], y['valid'], test_size=0.5, random_state=args.seed, stratify=y['valid'])

    print('Number of examples: ', len(examples['train']), len(examples['valid']), len(examples['test']))

    edge_label = {}
    edge_label_index = {}
    for mode in modes:
        examples[mode] = [[nodes.index(pair[0]), nodes.index(pair[1])] for pair in examples[mode]]
        edge_label[mode] = torch.FloatTensor(y[mode]) # to device
        edge_label_index[mode]=torch.tensor([examples[mode]]).permute(2, 1, 0)

    # prepare model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ckpt_name = f'ckpt/{args.conv}{args.nlayers}_{args.comb_type}_{args.seed}'

    # contrastive pretraining
    if args.train_mode == 'contra':
        print('Pre-train with contrastive loss...')
        contra_model = CombGNN(args.conv, args.nlayers, pyg_graph.num_nodes, hidden_dim=128, output_dim=1, comb_type=args.comb_type).to(device)
        contra_optimizer = torch.optim.Adam(contra_model.parameters(), lr=args.contra_lr, weight_decay=args.weight_decay)
        contra_early_stopping = EarlyStopping(patience=20, verbose=True, path=f'{ckpt_name}_contra.pt')

        for epoch in range(args.epochs):
            train_loss = train_contrastive(contra_model, device, pyg_graph, edge_label_index['train'], edge_label['train'], contra_optimizer)
            print(f'Contra Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f}')
            contra_early_stopping(train_loss, contra_model)
            if contra_early_stopping.early_stop:
                print('Early stopping')
                break

        del contra_model
        del contra_optimizer
        del contra_early_stopping
        torch.cuda.empty_cache()

    # cross entropy training
    print("Train with cross entropy loss...")
    model = CombGNN(args.conv, args.nlayers, pyg_graph.num_nodes, hidden_dim=128, output_dim=1, comb_type=args.comb_type).to(device)
    print("Model architecture: ")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.ce_lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    if args.train_mode == 'contra':
        model.load_state_dict(torch.load(f'{ckpt_name}_contra.pt'))
    
    early_stopping = EarlyStopping(patience=20, verbose=True, path=f'{ckpt_name}.pt')
    metric_list = [accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score]

    for epoch in range(args.epochs):
        train_loss, train_scores = train_ce(model, device, pyg_graph, edge_label_index['train'], edge_label['train'], criterion, optimizer, metric_list)
        valid_loss, valid_scores = evaluate(model, device, pyg_graph, edge_label_index['valid'], edge_label['valid'], criterion, metric_list)

        if args.wandb:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_scores[0],
                'valid_acc': valid_scores[0],
                'train_auc': train_scores[1],
                'valid_auc': valid_scores[1],
                'train_f1': train_scores[2],
                'valid_f1': valid_scores[2],
                'train_ap': train_scores[3],
                'valid_ap': valid_scores[3],
                'train_precision': train_scores[4],
                'valid_precision': valid_scores[4],
                'train_recall': train_scores[5],
                'valid_recall': valid_scores[5],
            })

        print(f'Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f} | Train Acc: {train_scores[0]*100:.2f}% | Train Precision: {train_scores[4]:.4f} | Train Recall: {train_scores[5]:.4f} || Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_scores[0]*100:.2f}% | Valid Precision: {valid_scores[4]:.4f} | Valid Recall: {valid_scores[5]:.4f}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    
    test_loss, test_scores = evaluate(model, device, pyg_graph, edge_label_index['test'], edge_label['test'], criterion, metric_list, checkpoint=f'{ckpt_name}.pt')
    if args.wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_scores[0],
            'test_auc': test_scores[1],
            'test_f1': test_scores[2],
            'test_ap': test_scores[3],
            'test_precision': test_scores[4],
            'test_recall': test_scores[5],
        })

    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_scores[0]*100:.2f}% | Test Precision: {test_scores[4]:.4f} | Test Recall: {test_scores[5]:.4f}')

if __name__ == '__main__':
    main()