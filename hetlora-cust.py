import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import (
    AutoModelForCausalLM, 
    LlamaForCausalLM, 
    GPT2LMHeadModel, 
    LlavaForConditionalGeneration,
    AutoProcessor
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
from fed_utils import (
    Aggregate, 
    FedAvg, 
    client_selection, 
    global_evaluation, 
    BroadCast, 
    GeneralClient, 
    VLLMClient,
    NORMAL_ALGO,
    compute_model_l2_norm,
)
import copy
from datasetter.llava import LlavaStyleDataset
import gc
from otherconfig import cuda_allocations, dataset_test_format
from evaluation.llava import llava_evaluation
import random
import numpy as np

def set_random_seed(seed: int = 42):
    random.seed(seed)              
    np.random.seed(seed)                
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)             
    torch.cuda.manual_seed_all(seed)         



def fl_finetune(
        # model/data params
        global_model: str = 'huggyllama/llama-7b',
        data_path: str = 'data_recaps/',
        output_dir: str = './fedgpt-llama7b-5-2/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 5,
        num_clients: int = 2,
        # Local training hyperparams
        local_batch_size: int = 32,  # 64,
        local_micro_batch_size: int = 2,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        # lora_r: int = 16,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter # The prompt template to use, will default to alpaca.
        # heterogeneous
        heter: bool = False,
        local_ranks: List[int] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        dataset_name = 'recap',
        mini: bool = True,
        algo: str = 'Hetlora',
        missing_rate:float = 0.4,
        file_format:str = '',
        cuda_idx:int = 0,
        train_subrate:float = 0.5,
        test_parts:int = 3,
        per_test:bool = False,
        lora_key:str = 'lora_A',
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"dataset: {dataset_name or False}\n"
            f"mini: {mini or False}\n"
            f"cuda_allocation: {cuda_allocations[cuda_idx]}\n"
            f"algo: {algo}\n"
            f"missing_rate: {missing_rate}\n"
            f"local_ranks: {local_ranks}\n"
            f"train_subrat:{train_subrate}\n"
            f"test_parts:{test_parts}\n"
            f"is personalized test:{per_test}\n"
            f"lora_key:{lora_key}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_model='decapoda-research/llama-7b-hf'"
    set_random_seed(666)
    # We need to set the max_memory for each device in order to make use of different devices.
    # max_memory = {0:'21GiB',1:"23GiB",2:'23GiB',3:'23GiB'}
    print('--lora_target_modules--:',lora_target_modules)
    max_memory = cuda_allocations[cuda_idx]
    
    # This args is for the special file format: for example recaps_0.json
    file_format = file_format + '{}.json' if not mini else file_format + '{}_mini.json'
    
    # Input data_path
    data_path = os.path.join(data_path, str(num_clients))
    missing_dir = os.path.join(data_path,'missing',str(missing_rate))
    # data_path = os.path.join(data_path)
    # assert (os.path.exists(data_path), "Please generate the data files for each client")
    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token='your token',
        )
    elif global_model == 'llava7b':
        # model = LlavaForConditionalGeneration.from_pretrained(
        # model =  LlavaOnevisionForConditionalGeneration.from_pretrained(
        #     # "bczhou/tiny-llava-v1-hf", 
        #     "visheratin/MC-LLaVA-3b",
        #     # "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        #     torch_dtype=torch.float16, 
        #     device_map=device_map,
        #     max_memory = {0:'22GiB',2:'22GiB',3:'22GiB'},
        #     # device_ids=[1, 2, 3],
        # )
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            # repo_type = "local",
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory = max_memory if not mini else None,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token="your token",
        )
    
    if global_model == 'llava7b':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        # processor = AutoProcessor.from_pretrained(
        #     "bczhou/tiny-llava-v1-hf",
        #     # patch_size = 32,
        #     # vision_feature_select_strategy='uniform',
        # )
        # processor = AutoProcessor.from_pretrained('llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
        # config = processor.config
        # config.save_pretrained('./myconfig')
        # processor.patch_size = 14
        # processor.vision_feature_select_strategy = "default"
        # print(processor.patch_size,processor.vision_feature_select_strategy)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    # image_processor.size = {"height": 378, "width": 378}
    processor.num_additional_image_tokens = 1
    #model = prepare_model_for_int8_training(model)
    # config = LoraConfig(
    #     # base_model_name_or_path="bczhou/tiny-llava-v1-hf",
    #     base_model_name_or_path="llava-hf/llava-1.5-7b-hf",
    #     r = lora_r,
    #     lora_alpha = lora_alpha,
    #     target_modules = lora_target_modules,
    #     lora_dropout = lora_dropout,
    #     bias = "none",
    #     task_type = "CAUSAL_LM",
    # )
    if algo == 'flora':
        global_config = LoraConfig(
            base_model_name_or_path="llava-hf/llava-1.5-7b-hf",
            r = lora_r * num_clients,
            lora_alpha = lora_alpha,
            target_modules = lora_target_modules,
            lora_dropout = lora_dropout,
            bias = "none",
            task_type = "CAUSAL_LM",
        )
        g_model = model
    elif algo.lower() in NORMAL_ALGO:
        global_config = LoraConfig(
            base_model_name_or_path="llava-hf/llava-1.5-7b-hf",
            r = max(local_ranks),
            # lora_alpha = lora_alpha * max(local_ranks),
            lora_alpha = max(local_ranks) * 2,
            target_modules = lora_target_modules,
            lora_dropout = lora_dropout,
            bias = "none",
            task_type = "CAUSAL_LM",
        )
        g_model = get_peft_model(model, global_config)
    
    if not ddp and torch.cuda.device_count() > 1:
        g_model.is_parallelizable = True
        g_model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
    client_list = []
    # for i in range(num_clients):
    #     client_list.append(VLLMClient(i, None, os.path.join(f'/home/lishan/workspace/FederatedLLM/data_recaps/'), output_dir)) 
    #     client_list[i].preprare_local_dataset(
    #         tokenizer=tokenizer,
    #         image_processor = image_processor,
    #     )
        

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)
        
        # c_model = g_model
        print(f'---- global before train norm2: {compute_model_l2_norm(g_model)}')
        personalized_scores = {}
        for client_id in selected_clients_set:
            # if algo == 'flora':
            #     c_model = copy.deepcopy(model)
            # elif algo == 'hetlora':
            if mini:
                c_model = model
            else:
                c_model = copy.deepcopy(model)
            # client = client_list[client_id]
            client = VLLMClient(
                client_id = client_id, 
                data_path = data_path,
                output_dir = output_dir,
                missing_dir = missing_dir,
            )
            client.preprare_local_dataset(
                # tokenizer=tokenizer,
                # image_processor = image_processor,
                processor = processor,
                dataset_name = file_format,
                train_subrate = train_subrate,
            )
            if heter:
                crank = local_ranks[client_id]
            #     clora_alpha = 2*local_ranks[client_id]
            # else:
            #     crank = lora_r
            clora_alpha = lora_alpha
            # set a config and assign a variable to it.
            config = LoraConfig(
                r=crank,
                lora_alpha=clora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                    # base_model_name_or_path=global_model,
                base_model_name_or_path = "llava-hf/llava-1.5-7b-hf",
            )
                # model_client = copy.deepcopy(model)
            # if epoch == 0 or algo == 'flora':
            c_model = get_peft_model(c_model, config)
            if epoch > 0 and algo != 'flora':
                BroadCast(c_model, client_id, output_dir, epoch, crank)
            total_params = sum(p.numel() for p in c_model.parameters())
            trainable_params = sum(p.numel() for p in c_model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable ratio: {trainable_params / total_params:.2%}")

            # 计算大小 (假设 float16)
            param_size_bytes = trainable_params * 2
            param_size_mb = param_size_bytes / (1024 ** 2)
            print(f"Trainable parameter size: {param_size_mb:.2f} MB")
            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            # client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            # client.build_local_trainer(
            #     tokenizer,
            #     local_micro_batch_size,
            #     gradient_accumulation_steps,
            #     local_num_epochs,
            #     local_learning_rate,
            #     group_by_length,
            #     ddp
            # )
            client.model = c_model
            print(f'{client_id} Before training, L2 Norm:',compute_model_l2_norm(client.model))

            #     epoch = epoch, 
            #     client_id = client_id, 
            #     model = client.model, 
            #     processor = processor, 
            #     tokenizer = tokenizer, 
            #     image_processor = image_processor, 
            #     data_path = data_path, 
            #     file_format=file_format
            # )
            
            client.build_local_trainer(
                processor,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp
            )
            
            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()
            print(f'{client_id} Before training, L2 Norm:',compute_model_l2_norm(client.model))
            # print(f"{client_id}",acc)
            print("Local training starts ... ")
            client.train()
            print(f'{client_id} After Trained---L2 Norm:{compute_model_l2_norm(client.model)}')
            #     epoch = epoch, 
            #     client_id = client_id, 
            #     model = client.model, 
            #     processor = processor, 
            #     tokenizer = tokenizer, 
            #     image_processor = image_processor, 
            #     data_path = data_path, 
            #     file_format = file_format
            
            if algo.lower() == 'fedilora':
                client.global_align(epoch,local_ranks[client_id],lora_key)
            if epoch >= 6:
                test_client_id = client_id if dataset_name not in dataset_test_format else dataset_test_format[dataset_name]
                acc = llava_evaluation(
                    epoch = epoch, 
                    client_id = test_client_id, 
                    model = client.model, 
                    processor = processor, 
                    tokenizer = tokenizer, 
                    image_processor = image_processor, 
                    data_path = data_path, 
                    file_format=file_format,
                    test_parts = test_parts, # this args is for partition the test set.
                    output_dir = output_dir,
                )
                print(f'client evaluation {client_id}:{acc}')
                acc = acc[1]
                personalized_scores = {key:personalized_scores.get(key,0) + acc[key] for key in acc}
            print("\nTerminating the local training of Client_{}".format(client_id))
            c_model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client
            gc.collect()
        print("Collecting the weights of clients and performing aggregation")
        #local_dataset_len_dict = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
        if epoch >= 6:
            for key in personalized_scores:
                personalized_scores[key] /= len(selected_clients_set)
            print(f'personalized score:{personalized_scores}')
        # g_model = FedAvg(
        #     g_model, 
        #     selected_clients_set,
        #     output_dir,
        #     local_dataset_len_dict,
        #     epoch,
        #     stacking,
        #     lora_r,
        #     heter,
        #     local_ranks,
        #     zero_padding,
        #     full
        # )
        # g_model = 
        Aggregate(
            model = g_model,
            selected_clients_set = selected_clients_set,
            output_dir = output_dir,
            local_dataset_len_dict = local_dataset_len_dict,
            epoch = epoch,
            algo = algo,
            lora_r = lora_r,
            local_ranks = local_ranks,
            heter = heter,
        )
        print(f'---- global after train norm2: {compute_model_l2_norm(g_model)}')
        '''<<<<<<Temp commented'''
        # pretrain_save_dir is for server and global model
        pretrain_save_dir = os.path.join(output_dir, str(epoch))
        if algo.lower() == 'flora':
            # This config_ori is for saving purpose
            ranks = 0
            for i in selected_clients_set:
                ranks += local_ranks[i]
            global_config.r = ranks
            global_config.lora_alpha = lora_alpha
            global_config.save_pretrained(
                pretrain_save_dir,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            # This pretrain_save_dir already has the adapter files.
            g_model = PeftModel.from_pretrained(model, pretrain_save_dir)
        # elif algo.lower() in 'hetlora':
        elif algo.lower() in NORMAL_ALGO:
            # Here is saving the lora parameters only.
            new_adapter_sd = torch.load(os.path.join(pretrain_save_dir,"adapter_model.bin"))
            set_peft_model_state_dict(g_model, new_adapter_sd, "default")
            global_config.save_pretrained(
                pretrain_save_dir,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            # adapter_bin = os.path.join(pretrain_save_dir,'adapter_model.bin')
            # g_model.load_adapter(adapter_bin, adapter_name = 'default', replace = True)
            # print(f'---- global load from pretrain path norm2: {compute_model_l2_norm(g_model)}')
            # g_model = PeftModel.from_pretrained(g_model, pretrain_save_dir)
            print(check_model_and_adapter(g_model,new_adapter_sd))
            # print(compute_model_l2_norm(get_peft_model_state_dict(g_model, adapter_name="default")))
            # adapter_params = {k: v for k, v in g_model.named_parameters() if "lora_" in k}
            # norm = torch.norm(torch.stack([p.norm() for p in adapter_params.values()]))
            # print(f"Adapter norm2 after loading new weights: {norm.item():.2f}")
        # print(f'---- global load from pretrain path norm2: {compute_model_l2_norm(g_model)}')
        #     config = AutoConfig.from_pretrained(global_model)
        #     tokenizer.save_pretrained(os.path.join(output_dir, str(epoch)),
        #             load_in_8bit=False,
        #             torch_dtype=torch.float32,
        #             device_map=device_map,)
        #     config.save_pretrained(os.path.join(output_dir, str(epoch)),
        #             load_in_8bit=False,
        #             torch_dtype=torch.float32,
        #             device_map=device_map,)

            # print('save model')
        
        # acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        
        '''>>>>>>Temp commented'''
        '''x_dir = os.path.join(output_dir, str(epoch))
        current_dir = x_dir # + "/temp/"
        print(current_dir)'''
        # arc_easy,hellaswag,mmlu,truthfulqa 
        # os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = current_dir))
        # os.system("lm_eval --model_args pretrained={current_dir},parallelize=True,load_in_4bit=False --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
        '''<<<<<<<<Temp commented'''
        if algo == 'flora':
            g_model = g_model.merge_and_unload()
            g_model.save_pretrained(
                os.path.join(output_dir, str(epoch) + '/final'),
                load_in_8bit=False,
                torch_dtype=torch.float32,
                device_map=device_map
            )
        
        if epoch >= 6:
            test_client_id = client_id if dataset_name not in dataset_test_format else dataset_test_format[dataset_name]
            acc = llava_evaluation(
                epoch = epoch, 
                client_id = test_client_id, 
                model = g_model, 
                processor = processor, 
                tokenizer = tokenizer, 
                image_processor = image_processor, 
                data_path = data_path, 
                file_format = file_format,
                test_parts = test_parts, # this args is for partition the test set.
                output_dir = output_dir,
            )
            print('Acc of Epoch', str(epoch), 'is:', acc)
        else:
            acc = 'no global test'
        acc_list.append(acc)
        
        if epoch < (num_communication_rounds - 1):
            if epoch - 1 >= 0:
                rm_dir = os.path.join(output_dir, str(epoch - 1))
                os.system("rm -rf {xxxxx}".format(xxxxx = rm_dir))
        '''>>>>>>>Temp commented'''
    print(acc_list)          
    #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'log.txt'
    file = open(filename,'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Log Saved")

def check_model_and_adapter(g_model,adapter):
    # this work
    total_norm = 0
    for key,param in get_peft_model_state_dict(g_model, adapter_name="default").items():
        if key in adapter:
            total_norm += torch.norm(param, p=2).item() ** 2  # accumulation square
    return total_norm ** 0.5
    # it doesn't work
    # for key,param in g_model.state_dict().items():
    #     if key in adapter:
    #         print(f'{key} is in g_model')
    

if __name__ == "__main__":
    fire.Fire(fl_finetune)