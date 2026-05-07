import torch
import os

from peft import set_peft_model_state_dict
import torch.nn.functional as F
import numpy as np
import time
import json
# set_peft_model_state_dict(model, lora_state_dict, adapter_name="default")

LORA_A = 'lora_A'
LORA_B = 'lora_B'
LORA = 'lora'

class FedServer:
    def __init__(self):
        # Store the lora parameters for each layers.
        self.weighted_single_weights = {}
    
    def compute_sim(self, client):
        '''
        weighted_single_weights: Store the parameter for each lora parameter {layer_0_lora_A: params}
        '''
        for key in self.weighted_single_weights:
            pass

def BroadCast(client_model,client_idx,output_dir, epoch,rank_num,global_lora = None):
    # if isinstance(ranks,list):
    #     rank_num = ranks[client_idx]
    # else:
    #     rank_num = ranks
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    lora_dict = torch.load(lora_path,map_location='cpu')
    # client_model.
    
    for key,param in lora_dict.items():
        if param.shape[0] > param.shape[1]:
            lora_dict[key] = param[:,:rank_num]
        else:
            lora_dict[key] = param[:rank_num,:]
    set_peft_model_state_dict(client_model, lora_dict, adapter_name = 'default')

def GlobalAlignmentByNums(client_model,output_dir,epoch,rank_num,replace_key = LORA_A,client_id = 0,edit_nums = 1):
    if epoch - 1 < 0 or replace_key.find('lora') < 0:
        return
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    global_lora_dict = torch.load(lora_path,map_location='cpu')
    
    local_lora_dict = client_model.state_dict()
    sim_collections = []
    layers_name = []
    min_sim_global_param = None
    min_sim = 2.0
    min_layer = ''
    sim_summary = []
    for key,local_params in local_lora_dict.items():
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        layer_sim =  F.cosine_similarity(local_params.view(-1), global_parm.view(-1), dim=0).mean().item()
        sim_collections.append(layer_sim)
        # max_sim = max(max_sim,layer_sim)
        layers_name.append(key)
        sim_summary.append((layer_sim,key))
        if key.find(replace_key) >= 0 and layer_sim < min_sim:
            min_sim = layer_sim
            min_layer = key
        # min_sim = min(min_sim,layer_sim)
        # local_lora_dict[key].data.copy_(global_parm.data * (1 - layer_sim) + local_params.data * layer_sim)
    agmax,agmin = np.argmax(sim_collections),np.argmin(sim_collections)
    layer_sim_dict = {layers_name[i] : sim_collections[i] for i in range(len(layers_name))}
    layer_sim_dict['replaced_key'] = replace_key
    
    sim_summary.sort(key = lambda x:x[0])
    for i in range(edit_nums):
        key = sim_summary[i][1]
        sim = sim_summary[i][0]
        local_params = local_lora_dict[key]
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        local_lora_dict[key].data.copy_(global_parm * (1 - sim) + local_params * sim)
    
    new_epoch_dir = os.path.join(output_dir,str(epoch))
    if not os.path.exists(new_epoch_dir):
        os.makedirs(new_epoch_dir,exist_ok=True)
    sim_save_dir = os.path.join(new_epoch_dir,f'sim_{client_id}.json')
    json.dump(layer_sim_dict,open(sim_save_dir,'w'))
    # global_param = global_lora_dict[layers_name[agmin]]
    print(f'Smallest layer of Lora_A:{min_layer}, and with similarity:{min_sim}')
    print(f'Mean of similarity:{np.mean(sim_collections)}, Median:{np.median(sim_collections)}, \
        Std:{np.std(sim_collections)}\n Max similarity:{np.max(sim_collections)} from {layers_name[agmax]} \nMin:{np.min(sim_collections)} from {layers_name[agmin]}')


def GlobalAlignment(client_model,output_dir,epoch,rank_num,replace_key = LORA_A,client_id = 0):
    if epoch - 1 < 0 or replace_key.find('lora') < 0:
        return
    start_time = time.time()
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    global_lora_dict = torch.load(lora_path,map_location='cpu')
    
    local_lora_dict = client_model.state_dict()
    sim_collections = []
    layers_name = []
    min_sim_global_param = None
    min_sim = 2.0
    min_layer = ''
    for key,local_params in local_lora_dict.items():
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        layer_sim =  F.cosine_similarity(local_params.view(-1), global_parm.view(-1), dim=0).mean().item()
        sim_collections.append(layer_sim)
        # max_sim = max(max_sim,layer_sim)
        layers_name.append(key)
        if key.find(replace_key) >= 0 and layer_sim < min_sim:
            min_sim_global_param = global_parm
            min_sim = layer_sim
            min_layer = key
        # min_sim = min(min_sim,layer_sim)
        # local_lora_dict[key].data.copy_(global_parm.data * (1 - layer_sim) + local_params.data * layer_sim)
    agmax,agmin = np.argmax(sim_collections),np.argmin(sim_collections)
    layer_sim_dict = {layers_name[i] : sim_collections[i] for i in range(len(layers_name))}
    layer_sim_dict['replaced_key'] = replace_key
    
    new_epoch_dir = os.path.join(output_dir,str(epoch))
    if not os.path.exists(new_epoch_dir):
        os.makedirs(new_epoch_dir,exist_ok=True)
    sim_save_dir = os.path.join(new_epoch_dir,f'sim_{client_id}.json')
    json.dump(layer_sim_dict,open(sim_save_dir,'w'))
    # global_param = global_lora_dict[layers_name[agmin]]
    local_params = local_lora_dict[min_layer]
    # global_param = global_param.to(local_params.device)
    if min_sim >= 0.95:
        print('No need to replace it. To be continue...')
        return
    local_lora_dict[min_layer].data.copy_(min_sim_global_param * (1 - min_sim) + local_params * min_sim)
    end_time = time.time()
    print(f'align time cost:{end_time - start_time}')
    print(f'Smallest layer of Lora_A:{min_layer}, and with similarity:{min_sim}')
    print(f'Mean of similarity:{np.mean(sim_collections)}, Median:{np.median(sim_collections)}, \
        Std:{np.std(sim_collections)}\n Max similarity:{np.max(sim_collections)} from {layers_name[agmax]} \nMin:{np.min(sim_collections)} from {layers_name[agmin]}')


def GlobalAlignmentAll(client_model,output_dir,epoch,rank_num,replace_key = LORA_A,client_id = 0):
    if epoch - 1 < 0 or replace_key.find('lora') < 0:
        return
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    global_lora_dict = torch.load(lora_path,map_location='cpu')
    
    local_lora_dict = client_model.state_dict()
    sim_collections = []
    layers_name = []
    min_sim_global_param = None
    min_sim = 2.0
    min_layer = ''
    for key,local_params in local_lora_dict.items():
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        layer_sim =  F.cosine_similarity(local_params.view(-1), global_parm.view(-1), dim=0).mean().item()
        sim_collections.append(layer_sim)
        # max_sim = max(max_sim,layer_sim)
        layers_name.append(key)
        if key.find(replace_key) >= 0 and layer_sim < min_sim:
            min_sim_global_param = global_parm
            min_sim = layer_sim
            min_layer = key
        # min_sim = min(min_sim,layer_sim)
        # local_lora_dict[key].data.copy_(global_parm.data * (1 - layer_sim) + local_params.data * layer_sim)
    agmax,agmin = np.argmax(sim_collections),np.argmin(sim_collections)
    layer_sim_dict = {layers_name[i] : sim_collections[i] for i in range(len(layers_name))}
    layer_sim_dict['replaced_key'] = replace_key
    
    new_epoch_dir = os.path.join(output_dir,str(epoch))
    if not os.path.exists(new_epoch_dir):
        os.makedirs(new_epoch_dir,exist_ok=True)
    sim_save_dir = os.path.join(new_epoch_dir,f'sim_{client_id}.json')
    json.dump(layer_sim_dict,open(sim_save_dir,'w'))
    # global_param = global_lora_dict[layers_name[agmin]]
    local_params = local_lora_dict[min_layer]
    # global_param = global_param.to(local_params.device)
    local_lora_dict[min_layer].data.copy_(min_sim_global_param)
    print(f'Smallest layer of Lora_A:{min_layer}, and with similarity:{min_sim}')
    print(f'Mean of similarity:{np.mean(sim_collections)}, Median:{np.median(sim_collections)}, \
        Std:{np.std(sim_collections)}\n Max similarity:{np.max(sim_collections)} from {layers_name[agmax]} \nMin:{np.min(sim_collections)} from {layers_name[agmin]}')


def GlobalAlignmentHalf(client_model,output_dir,epoch,rank_num,replace_key = LORA_A,client_id = 0):
    if epoch - 1 < 0 or replace_key.find('lora') < 0:
        return
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    global_lora_dict = torch.load(lora_path,map_location='cpu')
    
    local_lora_dict = client_model.state_dict()
    sim_collections = []
    layers_name = []
    min_sim_global_param = None
    min_sim = 2.0
    min_layer = ''
    for key,local_params in local_lora_dict.items():
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        layer_sim =  F.cosine_similarity(local_params.view(-1), global_parm.view(-1), dim=0).mean().item()
        sim_collections.append(layer_sim)
        # max_sim = max(max_sim,layer_sim)
        layers_name.append(key)
        if key.find(replace_key) >= 0 and layer_sim < min_sim:
            min_sim_global_param = global_parm
            min_sim = layer_sim
            min_layer = key
        # min_sim = min(min_sim,layer_sim)
        # local_lora_dict[key].data.copy_(global_parm.data * (1 - layer_sim) + local_params.data * layer_sim)
    agmax,agmin = np.argmax(sim_collections),np.argmin(sim_collections)
    layer_sim_dict = {layers_name[i] : sim_collections[i] for i in range(len(layers_name))}
    layer_sim_dict['replaced_key'] = replace_key
    
    new_epoch_dir = os.path.join(output_dir,str(epoch))
    if not os.path.exists(new_epoch_dir):
        os.makedirs(new_epoch_dir,exist_ok=True)
    sim_save_dir = os.path.join(new_epoch_dir,f'sim_{client_id}.json')
    json.dump(layer_sim_dict,open(sim_save_dir,'w'))
    # global_param = global_lora_dict[layers_name[agmin]]
    local_params = local_lora_dict[min_layer]
    # global_param = global_param.to(local_params.device)
    local_lora_dict[min_layer].data.copy_(min_sim_global_param * 0.5 + local_params * 0.5)
    print(f'Smallest layer of Lora_A:{min_layer}, and with similarity:{min_sim}')
    print(f'Mean of similarity:{np.mean(sim_collections)}, Median:{np.median(sim_collections)}, \
        Std:{np.std(sim_collections)}\n Max similarity:{np.max(sim_collections)} from {layers_name[agmax]} \nMin:{np.min(sim_collections)} from {layers_name[agmin]}')


def GlobalAlignmentOld(client_model,output_dir,epoch,rank_num):
    if epoch - 1 < 0:
        return
    lora_path = os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")
    global_lora_dict = torch.load(lora_path,map_location='cpu')
    
    local_lora_dict = client_model.state_dict()
    sim_collections = []
    layers_name = []
    min_sim_global_param = None
    min_sim = 2.0
    for key,local_params in local_lora_dict.items():
        if local_params.shape[0] > local_params.shape[1]:
            global_parm = global_lora_dict[key][:,:rank_num]
            # layer_sim =  F.cosine_similarity(local_params, global_lora_dict[key][:,:rank_num], dim=0).item()
        else:
            global_parm = global_lora_dict[key][:rank_num,:]
        global_parm = global_parm.to(local_params.device)
        layer_sim =  F.cosine_similarity(local_params.view(-1), global_parm.view(-1), dim=0).mean().item()
        sim_collections.append(layer_sim)
        # max_sim = max(max_sim,layer_sim)
        layers_name.append(key)
        if layer_sim < min_sim:
            min_sim_global_param = global_parm
            min_sim = layer_sim
        # min_sim = min(min_sim,layer_sim)
        # local_lora_dict[key].data.copy_(global_parm.data * (1 - layer_sim) + local_params.data * layer_sim)
    agmax,agmin = np.argmax(sim_collections),np.argmin(sim_collections)
    
    # global_param = global_lora_dict[layers_name[agmin]]
    local_params = local_lora_dict[layers_name[agmin]]
    # global_param = global_param.to(local_params.device)
    local_lora_dict[layers_name[agmin]].data.copy_(min_sim_global_param * (1-sim_collections[agmin]) + local_params * sim_collections[agmin])
    
    print(f'Mean of similarity:{np.mean(sim_collections)}, Median:{np.median(sim_collections)}, \
        Std:{np.std(sim_collections)}\n Max similarity:{np.max(sim_collections)} from {layers_name[agmax]} \nMin:{np.min(sim_collections)} from {layers_name[agmin]}')