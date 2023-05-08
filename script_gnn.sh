# Example command
## conv: ["GCN", "SAGE", "GAT", "GIN"]
## neg_dataset: ["TWOSIDES", "random"]
## neg_ratio: {1..3}
## seed: {0..9}
## train_mode: ["contra", "nocontra"]

python train_gnn.py --database "DC_combined" --comb_type "prod_fc" --conv "GIN" --neg_dataset "TWOSIDES" --train_mode "nocontra" --nlayers 2 --neg_ratio 1 --ce_lr 1e-3 --contra_lr 1e-2 --batch_size 32 --seed 0