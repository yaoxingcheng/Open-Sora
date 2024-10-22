torchrun --standalone --nproc_per_node 4 scripts/train.py \
    configs/phyv/train_base.py --data-path ../data/phyre/meta_info_cond.csv --flash-attn True