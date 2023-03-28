for neg_dataset in "random" "TWOSIDES"
do
    for embeddingf in "node2vec" "edge2vec" "res2vec_hetero" "res2vec_homo" "DREAMwalk"
    do
        for train_mode in "contra" "nocontra"
        do
            for seed in {0..9}
            do
            python train_rw.py --database "DC_combined" --comb_type "prod_fc" --embeddingf $embeddingf --neg_dataset $neg_dataset --batch_size 128 --neg_ratio 1 --ce_lr 1e-3 --contra_lr 1e-1 --train_mode $train_mode --seed $seed
            done
        done
    done
done