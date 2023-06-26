# Example command
## kgfeat: ["node2vec", "edge2vec", "res2vec_hetero", "res2vec_homo", "DREAMwalk"]
## chemfeat: ["ecfp", "maccs", "mordred"]
## neg_dataset: ["random", "TWOSIDES"]
## neg_ratio: {1..3}
## seed: {0..9}
## train_mode: ["contra", "nocontra"]
## wandb: whether to track with wandb or not

python train_rw.py --database "DC_combined_small" --comb_type "prod_fc" --kgfeat "DREAMwalk" --neg_dataset "TWOSIDES" --train_mode "nocontra" --neg_ratio 1 --batch_size 128 --ce_lr 1e-3 --contra_lr 1e-1 --seed 0 --device 1

# for neg_dataset in "TWOSIDES" "random"
# do
#     for kgfeat in "DREAMwalk" "node2vec"
#     do
#         for seed in {0..9}
#         do
#             python train_rw.py --database "DC_combined_small" --comb_type "prod_fc" --kgfeat $kgfeat --neg_dataset $neg_dataset --train_mode "nocontra" --neg_ratio 1 --batch_size 128 --ce_lr 1e-3 --contra_lr 1e-1 --seed $seed --device 1 --wandb
#         done
#     done
    
#     for chemfeat in "ecfp" "maccs" "mordred"
#     do
#         for seed in {0..9}
#         do
#             python train_rw.py --database "DC_combined_small" --comb_type "prod_fc" --chemfeat $chemfeat --neg_dataset $neg_dataset --train_mode "nocontra" --neg_ratio 1 --batch_size 128 --ce_lr 1e-3 --contra_lr 1e-1 --seed $seed --device 1 --wandb
#         done
#     done
    
#     for chemfeat in "ecfp" "maccs" "mordred"
#     do
#         for seed in {0..9}
#         do
#             python train_rw.py --database "DC_combined_small" --comb_type "prod_fc" --kgfeat "DREAMwalk" --chemfeat $chemfeat --neg_dataset $neg_dataset --train_mode "nocontra" --neg_ratio 1 --batch_size 128 --ce_lr 1e-3 --contra_lr 1e-1 --seed $seed --device 1 --wandb
#         done
#     done
# done