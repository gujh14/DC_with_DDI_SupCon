# Example command
## embeddingf: ["node2vec", "edge2vec", "res2vec_hetero", "res2vec_homo", "DREAMwalk"]
## neg_dataset: ["TWOSIDES", "random"]
## neg_ratio: {1..3}
## seed: {0..9}
## train_mode: ["contra", "nocontra"]

python train_rw.py --database "DC_combined" --comb_type "prod_fc" --embeddingf "DREAMwalk" --neg_dataset "TWOSIDES" --train_mode "contra" --neg_ratio 1 --batch_size 128 --ce_lr 1e-3 --contra_lr 1e-1 --seed 0