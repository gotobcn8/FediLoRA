#!/bin/bash
# recaps
# 0.6 missing

# fedilora
nohup python vllm.py \
    --algo fedilora \
    --global_model llava7b \
    --data_path data_recaps \
    --output_dir llava-recaps-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --cuda_idx 0 \
    --train_subrate 0.3 \
    --test_parts 5 > recaps_0.4_fedilora.log 2>&1 &


# hetlora
nohup python vllm.py \
    --algo hetlora \
    --global_model llava7b \
    --data_path data_recaps \
    --output_dir llava-recaps-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --cuda_idx 0 \
    --train_subrate 0.3 \
    --test_parts 5 > recaps_0.4_hetlora.log 2>&1 &

# flora
nohup python vllm.py \
    --algo flora \
    --global_model llava7b \
    --data_path data_recaps \
    --output_dir llava-recaps-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --cuda_idx 0 \
    --train_subrate 0.3 \
    --test_parts 5 > recaps_0.4_flora.log 2>&1 &

# 0.3 test one
nohup python vllm.py \
    --global_model llava7b \
    --data_path data_recaps \
    --output_dir llava-recaps-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.6 \
    --client_selection_frac 0.5 \
    --file_format recap_ \
    --cuda_idx 0 > recaps_0.4_fedilora.log 2>&1 &