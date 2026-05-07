import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from datasetter.llava import LlavaStyleDataset, TrainLLavaModelCollator
from torch.utils.data import Subset
from .server import (
    GlobalAlignment,
    GlobalAlignmentAll,
    GlobalAlignmentHalf,
    GlobalAlignmentByNums,
)
import random
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import json

class VLLMClient:
    def __init__(self, client_id, data_path, output_dir, missing_dir = None, model = None):
        self.client_id = client_id
        self.model = model
        self.data_path = data_path
        self.train_path = os.path.join(data_path,'train')
        self.test_path = os.path.join(data_path,'test')
        # self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        # self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.missing_file = os.path.join(missing_dir,f'{self.client_id}.json')
        self.missing_indexs = {}
        with open(self.missing_file,'r') as f:
            self.missing_indexs = json.load(f)
        
    def preprare_local_dataset(self, processor, dataset_name, train_subrate):
        self.train_dataset = LlavaStyleDataset(
            os.path.join(self.train_path, dataset_name.format(self.client_id)),
            # tokenizer=tokenizer,
            # image_processor = image_processor,
            max_length=512,
            processor = processor,
            missing_indexs = self.missing_indexs,
            data_subset_rate = train_subrate,
        )
        # self.test_dataset = LlavaStyleDataset(
        #     os.path.join(self.test_path, dataset_name.format(self.client_id)),
        #     # tokenizer=tokenizer,
        #     # image_processor = image_processor,
        #     processor = processor,
        # )
        # self.local_val_set_size = len(self.test_dataset)
        
    def build_local_trainer(self,
                            processor,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=3,
            optim="adamw_torch",
            # optim = 'adamw_torch_4bit',
            # optim = "sgd",
            # evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            # save_strategy="steps",
            # eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            # load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            remove_unused_columns = False,
        )
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        selected_indices = indices[:int(0.5 * len(indices))]
        sub_train_dataset = Subset(self.train_dataset, selected_indices)
        
        test_indices = list(range(len(self.train_dataset)))
        random.shuffle(test_indices)
        # test_selected_indices = indices[:int(0.5 * len(test_indices))]
        # sub_test_dataset = Subset(self.test_dataset, test_selected_indices)
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=sub_train_dataset,
            # eval_dataset=sub_test_dataset,
            args=self.train_args,
            data_collator = TrainLLavaModelCollator(processor,self.missing_indexs)
        )
        # breakpoint()

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        ### TODO it's lack of collator
        self.local_trainer.train()

    def global_align_by_nums(self,epoch,rank_num, lora_key = 'lora_A',edit_nums = 1):
        GlobalAlignmentByNums(
            client_model = self.model,
            output_dir = self.output_dir,
            epoch = epoch,
            rank_num = rank_num,
            replace_key = lora_key,
            client_id = self.client_id,
            edit_nums = edit_nums,
        )
    
    def global_align(self,epoch,rank_num, lora_key = 'lora_A'):
        GlobalAlignment(
            client_model = self.model,
            output_dir = self.output_dir,
            epoch = epoch,
            rank_num = rank_num,
            replace_key = lora_key,
            client_id = self.client_id
        )
    
    def global_align_all(self,epoch,rank_num, lora_key = 'lora_A'):
        GlobalAlignmentAll(
            client_model = self.model,
            output_dir = self.output_dir,
            epoch = epoch,
            rank_num = rank_num,
            replace_key = lora_key,
            client_id = self.client_id
        )
    
    def global_align_half(self,epoch,rank_num, lora_key = 'lora_A'):
        GlobalAlignmentHalf(
            client_model = self.model,
            output_dir = self.output_dir,
            epoch = epoch,
            rank_num = rank_num,
            replace_key = lora_key,
            client_id = self.client_id
        )
    
    def compute_model_l2_norm(self):
        total_norm = 0.0
        for key, param in self.model.state_dict().items():
            total_norm += torch.norm(param, p=2).item() ** 2  # 累加平方
        return total_norm ** 0.5
    
    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.train_dataset)
        new_adapter_weight = self.model.state_dict()
        # for key,param in self.model.named_parameters():
        #     if 'lora'in key and param.requires_grad:
        #         print(param.abs().mean())
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
            "default" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
