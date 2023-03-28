import torch
import numpy as np
import os
import random

import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

from model import CombNetContrastive
from dataset import CombinationDatasetRW
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='CombNet_RW')
    parser.add_argument('--database', type=str, default='DC_combined', choices=['C_DCDB', 'DCDB', 'DC_combined'])
    parser.add_argument('--embeddingf', type=str, default='DREAMwalk', choices=['node2vec', 'edge2vec', 'res2vec_homo', 'res2vec_hetero', 'DREAMwalk'])
    parser.add_argument('--neg_dataset', type=str, default='TWOSIDES', choices=['random', 'TWOSIDES'])
    parser.add_argument('--neg_ratio', type=int, default=1, help='negative ratio')
    parser.add_argument('--comb_type', type=str, default='prod_fc', choices=['sum', 'cosine', 'prod_fc'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--ce_lr', type=float, default=1e-3, help='learning rate for cross entropy loss')
    parser.add_argument('--contra_lr', type=float, default=1e-1, help='learning rate for contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--device', type=int, default=0, help='device number')
    # parser.add_argument('--ckpt_name', type=str, default='default_DREAMwalk_prod_fc', help='checkpoint name')
    parser.add_argument('--train_mode', type=str, default='contra', choices=['contra', 'nocontra'])
    args = parser.parse_args()
    return args

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
    
def train_contrastive(model, device, train_loader, optimizer, args):
    '''
    Contrastive training with info nce loss
    '''
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        cos_sim = model.forward_contrastive(data)
        info_nce_loss = model.info_nce_loss(cos_sim, target)
        info_nce_loss.backward()
        optimizer.step()
        train_loss += info_nce_loss.item()
    
    return train_loss / len(train_loader)

def train_ce(model, device, train_loader, criterion, optimizer, metric_list=[accuracy_score]):
    '''
    Cross entropy training
    '''
    model.train()
    train_loss = 0

    target_list = []
    pred_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_list.append(torch.sigmoid(output).detach().cpu().numpy())
        target_list.append(target.long().detach().cpu().numpy())

    # metric
    scores = []
    for metric in metric_list:
        if (metric == roc_auc_score) or metric == average_precision_score:
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list)))
        else:
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list).round()))

    return train_loss / len(train_loader), scores

def evaluate(model, device, loader, criterion, metric_list=[accuracy_score], checkpoint=None):
    # evaluate
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.eval()
    eval_loss = 0
    target_list = []
    pred_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.float().to(device)
            output = model(data).view(-1)
            eval_loss += criterion(output, target).item()
            pred_list.append(torch.sigmoid(output).detach().cpu().numpy())
            target_list.append(target.long().detach().cpu().numpy())

    scores = []
    for metric in metric_list:
        if (metric == roc_auc_score) or metric == average_precision_score:
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list)))
        else:
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list).round()))
    return eval_loss / len(loader), scores

def main():
    args = parse_args()
    group = f"{args.train_mode}_{args.embeddingf}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})_celr({args.ce_lr})_contralr({args.contra_lr})"
    wandb.init(project="DC_DDI_contrastive_rw", group=group, entity="gujh14")
    wandb.config.update(args)
    wandb.run.name = f"{args.train_mode}_{args.embeddingf}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})_seed{args.seed}"
    wandb.run.save()
    print(args)

    seed_everything(args.seed)

    dataset = CombinationDatasetRW(args.database, args.embeddingf, args.neg_ratio, args.neg_dataset, args.seed)
    train_dataset, valid_dataset, test_dataset = dataset['train'], dataset['valid'], dataset['test']
    print(f'Number of train, valid, test: {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[0] // 2 # 128
    hidden_dim = input_dim
    output_dim = 1

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ckpt_name = f'ckpt/{args.embeddingf}_{args.comb_type}_{args.seed}'

    if args.train_mode == 'contra':
        print("Pre-train with contrastive loss")
        contra_model = CombNetContrastive(input_dim, hidden_dim, output_dim, args.comb_type).to(device)
        contra_optimizer = torch.optim.Adam(contra_model.parameters(), lr=args.contra_lr, weight_decay=args.weight_decay)
        contra_early_stopping = EarlyStopping(patience=20, verbose=True, path=f"{ckpt_name}_contra.pt")

        for epoch in range(args.epochs):
            train_loss = train_contrastive(contra_model, device, train_loader, contra_optimizer, args)
            print(f'Contra Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f}')
            contra_early_stopping(train_loss, contra_model)
            if contra_early_stopping.early_stop:
                print("Early stopping")
                break
        
        del contra_model
        del contra_optimizer
        del contra_early_stopping
        torch.cuda.empty_cache()

    print("Train with cross entropy loss")
    model = CombNetContrastive(input_dim, hidden_dim, output_dim, args.comb_type).to(device)
    print("Model architecture: ")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.ce_lr, weight_decay=args.weight_decay)
    if args.train_mode == 'contra':
        model.load_state_dict(torch.load(f"{ckpt_name}_contra.pt"))
    
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{ckpt_name}.pt")
    metric_list = [accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score]

    for epoch in range(args.epochs):
        
        train_loss, train_scores = train_ce(model, device, train_loader, criterion, optimizer, metric_list)
        valid_loss, valid_scores = evaluate(model, device, valid_loader, criterion, metric_list)

        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc": train_scores[0],
            "valid_acc": valid_scores[0],
            "train_auc": train_scores[1],
            "valid_auc": valid_scores[1],
            "train_f1": train_scores[2],
            "valid_f1": valid_scores[2],
            "train_ap": train_scores[3],
            "valid_ap": valid_scores[3],
            "train_precision": train_scores[4],
            "valid_precision": valid_scores[4],
            "train_recall": train_scores[5],
            "valid_recall": valid_scores[5],
        })

        print(f'Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f} | Train Acc: {train_scores[0]*100:.2f}% | Train Precision: {train_scores[4]:.4f} | Train Recall: {train_scores[5]:.4f} || Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_scores[0]*100:.2f}% | Valid Precision: {valid_scores[4]:.4f} | Valid Recall: {valid_scores[5]:.4f}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    test_loss, test_scores = evaluate(model, device, test_loader, criterion, metric_list, checkpoint=f"{ckpt_name}.pt")
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_scores[0],
        "test_auc": test_scores[1],
        "test_f1": test_scores[2],
        "test_ap": test_scores[3],
        "test_precision": test_scores[4],
        "test_recall": test_scores[5],
    })

    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_scores[0]*100:.2f}% | Test Precision: {test_scores[4]:.4f} | Test Recall: {test_scores[5]:.4f}')

if __name__ == '__main__':
    main()