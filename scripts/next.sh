#!/bin/bash
# missing 0.4
# fedilora 3 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-3gpu-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > next_0.4_fedilora_3_gpu.log 2>&1 &


# fedilora 4 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-4gpu-10/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 2\
    --num_clients 10 \
    --heter True \
    --algo fedilora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 1 > next_0.4_fedilora_4_gpu.log 2>&1 &


# -------------------------------------------------------- first machine --------------------------------------------------------
# flora 4 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-flora-10-0.4/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo flora \
    --mini False \
    --missing_rate 0.4 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > next_0.4_flora.log 2>&1 &

#hetlora 4 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-hetlora-10-0.4/ \
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
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 1 > next_0.4_hetlora.log 2>&1 &



# -------------------------------------------------------- second machine --------------------------------------------------------
# flora 4 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-flora-10-0.5/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo flora \
    --mini False \
    --missing_rate 0.5 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > next_0.5_flora.log 2>&1 &

#hetlora 4 gpu
nohup python vllm.py \
    --global_model llava7b \
    --data_path /workspace/2025EMNLP-FLME/input_next/data_recaps \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-hetlora-10-0.5/ \
    --num_communication_rounds 8 \
    --local_num_epochs 1 \
    --local_batch_size 32\
    --local_micro_batch_size 4\
    --num_clients 10 \
    --heter True \
    --algo hetlora \
    --mini False \
    --missing_rate 0.5 \
    --client_selection_frac 0.4 \
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 1 > next_0.5_hetlora.log 2>&1 &

nohup python extra06.py \
    --global_model llava7b \
    --data_path next \
    --file_format "" \
    --dataset_name next \
    --output_dir llava-next-hetlora-10-0.4/ \
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
    --train_subrate 0.5 \
    --test_parts 3 \
    --cuda_idx 0 > next_0.4_hetlora-global-per.log 2>&1 &