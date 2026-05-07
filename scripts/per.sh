#!/bin/bash
# next (hetlora, flora)
nohup python vllm.py \
    --global_model llava7b \
    --data_path next \
    --output_dir fedilora-next-0.6-per/ \
    --num_communication_rounds 5 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 5 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.6 \
    --client_selection_frac 0.4 \
    --dataset_name next \
    --train_subrate 0.7 \
    --test_parts 4 \
    --per_test True \
    --cuda_idx 0 > next_0.6_fedilora-per.log 2>&1 &

# recap (hetlora, flora)
nohup python vllm.py \
    --global_model llava7b \
    --data_path recaps \
    --output_dir fedilora-recap-0.6-per/ \
    --num_communication_rounds 5 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 5 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.6 \
    --client_selection_frac 0.4 \
    --dataset_name recap \
    --train_subrate 0.3 \
    --test_parts 10 \
    --per_test True \
    --cuda_idx 0 > recap_0.6_fedilora-per.log 2>&1 &
