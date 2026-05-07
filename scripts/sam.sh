#!/bin/bash
# missing 0.4
# fedilora
nohup python vllm.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir hetlora-sam-0.4/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo hetlora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --dataset_name sam \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > sam_0.4_hetlora.log 2>&1 &

nohup python vllm.py --global_model llava7b --data_path sam --output_dir llava-sam-0.4/ --num_communication_rounds 8 --local_num_epochs 1 --local_batch_size 32 --local_micro_batch_size 4 --num_clients 10 --heter True     --algo fedilora     --mini False     --missing_rate 0.4     --client_selection_frac 0.4     --dataset_name sam     --train_subrate 0.5     --test_parts 3 \

nohup python vllm.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir llava-sam-0.5/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.5 \
    --client_selection_frac 0.4 \
    --dataset_name sam \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > sam_0.5_fedilora.log 2>&1 &


nohup python vllm.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir llava-sam-0.5/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.5 \
    --client_selection_frac 0.4 \
    --dataset_name sam \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > sam_0.5_fedilora.log 2>&1 &

nohup python extra06.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir sam_fedilora_06_norm2/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.5 \
    --client_selection_frac 0.4 \
    --dataset_name sam \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > sam_fedilora_06_norm2.log 2>&1 &

# flora
nohup python vllm.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir llava-sam-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --algo flora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > sam_0.4_flora.log 2>&1 &

#hetlora
nohup python vllm.py \
    --global_model llava7b \
    --data_path sam \
    --output_dir llava-sam-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 2 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --algo hetlora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 1 > sam_0.4_hetlora.log 2>&1 &

