#!/bin/bash

python3 Train.py \
    --data DB15K \
    --gpu 2 \
    --num_epoch 5000 \
    --valid_epoch 10 \
    --hidden_dim 1024 \
    --lr 4e-4 \
    --dim 256 \
    --max_vis_token 16 \
    --max_txt_token 24 \
    --num_head 4 \
    --emb_dropout 0.9 \
    --vis_dropout 0.4 \
    --txt_dropout 0.1 \
    --num_layer_dec 2 \
    --num_neg_samples 16 \

