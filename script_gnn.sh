# contra GCN
for conv in "GCN"
do
    for neg_dataset in "random" "TWOSIDES"
    do
        for train_mode in "contra"
        do
            for ce_lr in 1e-2
            do
                for contra_lr in 1e-3
                do
                    for seed in {0..9}
                    do
                    python train_gnn.py --database "DC_combined" --comb_type "prod_fc" --conv $conv --neg_dataset $neg_dataset --ce_lr $ce_lr --contra_lr $contra_lr --train_mode $train_mode --seed $seed --device 1 --wandb
                    done
                done
            done
        done
    done
done

# contra GAT, SAGE, GIN
for conv in "GAT" "SAGE" "GIN"
do
    for neg_dataset in "random" "TWOSIDES"
    do
        for train_mode in "contra"
        do
            for ce_lr in 1e-3
            do
                for contra_lr in 1e-3
                do
                    for seed in {0..9}
                    do
                    python train_gnn.py --database "DC_combined" --comb_type "prod_fc" --conv $conv --neg_dataset $neg_dataset --ce_lr $ce_lr --contra_lr $contra_lr --train_mode $train_mode --seed $seed --device 1 --wandb
                    done
                done
            done
        done
    done
done

# nocontra GCN
for conv in "GCN"
do
    for neg_dataset in "random" "TWOSIDES"
    do
        for train_mode in "nocontra"
        do
            for ce_lr in 1e-2
            do
                for contra_lr in 0
                do
                    for seed in {0..9}
                    do
                    python train_gnn.py --database "DC_combined" --comb_type "prod_fc" --conv $conv --neg_dataset $neg_dataset --ce_lr $ce_lr --contra_lr $contra_lr --train_mode $train_mode --seed $seed --device 1 --wandb
                    done
                done
            done
        done
    done
done

# nocontra GAT, SAGE, GIN
for conv in "GAT" "SAGE" "GIN"
do
    for neg_dataset in "random" "TWOSIDES"
    do
        for train_mode in "nocontra"
        do
            for ce_lr in 1e-3
            do
                for contra_lr in 0
                do
                    for seed in {0..9}
                    do
                    python train_gnn.py --database "DC_combined" --comb_type "prod_fc" --conv $conv --neg_dataset $neg_dataset --ce_lr $ce_lr --contra_lr $contra_lr --train_mode $train_mode --seed $seed --device 1 --wandb
                    done
                done
            done
        done
    done
done