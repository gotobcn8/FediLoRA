#!/bin/bash

python vllm.py \
    --global_model llava7b \
    --data_path data_recaps \
    --output_dir llava-recaps-10/ \
    --num_communication_rounds 2 \
    --local_num_epochs 1 \
    --local_batch_size 32 \
    --local_micro_batch_size 2 \
    --num_clients 10 \
    --heter True \
    --algo hetlora \
    --mini False \
    --missing_rate 0.4 \
    --file_format recap_ \
    --train_subrate 0.1 \
    --test_parts 10 \
    --per_test True
