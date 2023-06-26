import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

class CombNetRW(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, comb_type='prod_fc', dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        if comb_type not in ['sum', 'cosine', 'prod_fc', 'concat']:
            raise ValueError('comb_type must be one of [sum, cosine, prod_fc, concat]')
        self.comb_type = comb_type
        self.tr = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
        fc_dim = hidden_dim * 2 if comb_type == 'concat' else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward_contrastive(self, data):
        drug1, drug2 = data[:, :self.input_dim], data[:, self.input_dim:]
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        # cosine similarity
        return self.cosine_similarity(drug1, drug2).unsqueeze(1)
    
    def forward(self, data):
        drug1, drug2 = data[:, :self.input_dim], data[:, self.input_dim:]
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        if self.comb_type == 'sum':
            comb = drug1 + drug2
            return self.fc(comb)
        elif self.comb_type == 'cosine':
            # cosine similarity
            return self.cosine_similarity(drug1, drug2).unsqueeze(1)
        elif self.comb_type == 'prod_fc':
            comb = drug1 * drug2
            return self.fc(comb)
        elif self.comb_type == 'concat':
            # order invariant concat
            comb1 = torch.cat((drug1, drug2), dim=1)
            comb2 = torch.cat((drug2, drug1), dim=1)
            pred1 = self.fc(comb1)
            pred2 = self.fc(comb2)
            return (pred1 + pred2) / 2
        
    def info_nce_loss(self, cos_sim, labels):
        # labels: (batch_size, 1) with values 0 or 1
        # cos_sim: (batch_size, 1)
        # find the position of the positive / negative sample
        pos_idx = labels == 1
        neg_idx = labels == 0
        # compute the loss for the positive samples
        pos_loss = torch.exp(cos_sim[pos_idx]).sum(dim=0)
        # compute the loss for the negative samples
        neg_loss = torch.exp(cos_sim[neg_idx]).sum(dim=0)
        # compute the total loss
        loss = -torch.log(pos_loss / (pos_loss + neg_loss))
        return loss
    
    def extract_trained_feature(self, data, normalize=False):
        drug1, drug2 = data[:, :self.input_dim], data[:, self.input_dim:]
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        if normalize:
            drug1 = F.normalize(drug1, dim=1)
            drug2 = F.normalize(drug2, dim=1)
        return drug1 * drug2
    
class CombGNN(torch.nn.Module):
    def __init__(self, convlayer, nlayers, num_nodes, hidden_dim, output_dim, comb_type='prod_fc', dropout=0.1):
        super().__init__()
        self.nlayers = nlayers
        self.comb_type = comb_type

        self.embedding = torch.nn.Embedding(num_nodes, hidden_dim)
        self.convs = torch.nn.ModuleList()
    
        if convlayer == 'GIN':
            for _ in range(nlayers):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))

        else:
            if convlayer == 'GCN':
                conv = GCNConv
            elif convlayer == 'SAGE':
                conv = SAGEConv
            elif convlayer == 'GAT':
                conv = GATConv

            for _ in range(nlayers):
                self.convs.append(conv(hidden_dim, hidden_dim))

        self.tr = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward_contrastive(self, x, edge_index, edge_label_index):
        x = self.embedding(x)
        
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        
        drug1, drug2 = x[edge_label_index[0]].squeeze(), x[edge_label_index[1]].squeeze()
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        
        return self.cosine_similarity(drug1, drug2).unsqueeze(1)
    
    def forward(self, x, edge_index, edge_label_index):
        x = self.embedding(x)

        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)

        drug1, drug2 = x[edge_label_index[0]].squeeze(), x[edge_label_index[1]].squeeze()
        drug1, drug2 = self.tr(drug1), self.tr(drug2)

        if self.comb_type == 'sum':
            comb = drug1 + drug2
            return self.fc(comb)
        elif self.comb_type == 'cosine':
            # cosine similarity
            return self.cosine_similarity(drug1, drug2).unsqueeze(1)
        elif self.comb_type == 'prod_fc':
            comb = drug1 * drug2
            return self.fc(comb)
        
    def info_nce_loss(self, cos_sim, labels):
        # labels: (batch_size, 1) with values 0 or 1
        # cos_sim: (batch_size, 1)
        # find the position of the positive / negative sample
        pos_idx = labels == 1
        neg_idx = labels == 0

        # compute the loss for the positive samples
        pos_loss = torch.exp(cos_sim[pos_idx]).sum(dim=0)
        # compute the loss for the negative samples
        neg_loss = torch.exp(cos_sim[neg_idx]).sum(dim=0)
        # compute the total loss
        loss = -torch.log(pos_loss / (pos_loss + neg_loss))
        return loss
    
    def extract_trained_feature(self, x, edge_index, edge_label_index, normalize=False):
        x = self.embedding(x)

        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)

        drug1, drug2 = x[edge_label_index[0]].squeeze(), x[edge_label_index[1]].squeeze()
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        if normalize:
            drug1 = F.normalize(drug1, dim=1)
            drug2 = F.normalize(drug2, dim=1)
        return drug1 * drug2