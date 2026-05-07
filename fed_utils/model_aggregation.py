from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d
from .other import NORMAL_ALGO

def FedEditor():
    
    pass

NAME_REFLECT = {
    'A.weight':'B.weight',
    'B.weight':'A.weight',
}

# def compute_F_norm_weight_by_key(output_dir,epoch,selected_clients_set):
#     weights_array = {}
#     for k, client_id in enumerate(selected_clients_set):
#         single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
#             "pytorch_model.bin")
#         single_weights = torch.load(single_output_dir, map_location = 'cpu')
#         visited = set()
#         for key in single_weights:
#             splits = key.split('lora_')
#             prefix,suffix = splits[0],splits[1]
#             if prefix in visited:
#                 continue
#             visited.add(prefix)
#             other_suffix = NAME_REFLECT[suffix]
#             if single_weights[key].shape[0] < single_weights[key].shape[1]:
#                 S = single_weights[prefix + 'lora_' + other_suffix] @ single_weights[key]
#             else:
#                 S = single_weights[key] @ single_weights[prefix + 'lora_' + other_suffix]  
#             frobenius_norm = torch.norm(S, p='fro')
#             if k == 0:
#                 weights_array[key] = []
#             if frobenius_norm == 0:
#                 frobenius_norm = 1e-6
#             weights_array[key].append(frobenius_norm)
    
#     for _,weights in weights_array.items():
#         weights = [w / sum(weights) for w in weights]
            
#     return weights_array

def compute_F_norm_weight(output_dir,epoch,selected_clients_set):
    weights_array = {}
    clients_frobenius_norm_all = 0
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
            "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        visited = set()
        frobenius_norm_all = 0
        single_client_fb = 0
        for key in single_weights:
            splits = key.split('lora_')
            prefix,suffix = splits[0],splits[1]
            if prefix in visited:
                continue
            visited.add(prefix)
            other_suffix = NAME_REFLECT[suffix]
            if single_weights[key].shape[0] < single_weights[key].shape[1]:
                S = single_weights[prefix + 'lora_' + other_suffix] @ single_weights[key]
            else:
                S = single_weights[key] @ single_weights[prefix + 'lora_' + other_suffix]  
            frobenius_norm_all += torch.norm(S, p='fro')
            clients_frobenius_norm_all += frobenius_norm_all
            single_client_fb += frobenius_norm_all
        weights_array[client_id] = single_client_fb
        # weights_array[client_id] = (max(frobenius_norm_all,1e-6))
    
    for key in weights_array:
        weights_array[key] /= clients_frobenius_norm_all
            
    return weights_array

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, stacking, lora_r, heter, local_ranks, zero_padding, full):
    # if not zero_padding:
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
            dtype=torch.float32
        ),
    p=1, dim=0)
    # elif zero_padding:
    #     weights_array = compute_F_norm_weight(
    #         output_dir = output_dir,epoch = epoch,selected_clients_set = selected_clients_set,
    #     )
    # print("Weights:", weights_array)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
            "pytorch_model.bin")
        # single_weights is a dict included all lora parameters. key is the parameter name for example layer1.loraA = tensor()
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        #print(single_weights)
        #print("y")
        x = 0
        if full:
            if k == 0:
                weighted_single_weights = single_weights
                for key in weighted_single_weights.keys():
                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
            else:
                for key in single_weights.keys():
                    weighted_single_weights[key] += single_weights[key] * (weights_array[k])
            
        else:
            if stacking:
                if zero_padding:
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[key][k])
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[key][k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[key][k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[key][k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                        
                else:
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                            #print(weighted_single_weights[key].shape)
                            if heter:
                                x += 1
                                if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                            else:
                                if weighted_single_weights[key].shape[0] == lora_r:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

                    else:
                        for key in single_weights.keys():
                            if heter:
                                x += 1
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)
                            else:
                                if single_weights[key].shape[0] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)
                            
                            if heter:
                                if single_weights[key].shape[1] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)
                            else:
                                if single_weights[key].shape[1] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)

            else:
                if zero_padding:
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                else:
                    if k == 0:
                        weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                            single_weights.keys()}
                    else:
                        weighted_sindgle_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                            for key in
                                            single_weights.keys()}


    if stacking:
        # Save adapter only, and return model still is original model
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        return model
    elif full:
        # Overload the weighted single weights back to base_model
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "pytorch_model.bin"))
        model.load_state_dict(weighted_single_weights)
        return model
    else:
        # This one is for normal weighted average. A_{sum} = A_1 * weight_1 + A_2 * weight_2, no need to save adapter, 'cuz nothing changed(shape)
        set_peft_model_state_dict(model, weighted_single_weights, "default")
        return model


def Flora(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, lora_r, heter, local_ranks):
    # Compute the weight for each client
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
            dtype=torch.float32
        ),
    p=1, dim=0)
    
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
            "pytorch_model.bin")
        # single_weights is a dict included all lora parameters. key is the parameter name for example layer1.loraA = tensor()
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        #print(single_weights)
        #print("y")
        x = 0
        
        if k == 0:
            weighted_single_weights = single_weights
            for key in weighted_single_weights.keys():
                            #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                            #print(weighted_single_weights[key].shape)
                if heter:
                    x += 1
                    if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                else:
                    if weighted_single_weights[key].shape[0] == lora_r:
                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

        else:
            for key in single_weights.keys():
                if heter:
                    x += 1
                    if single_weights[key].shape[0] == local_ranks[client_id]:
                        new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                        weighted_single_weights[key] = torch.cat(new, dim=0)
                    else:
                        new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                        weighted_single_weights[key] = torch.cat(new, dim=1)
                else:
                    if single_weights[key].shape[0] == lora_r:
                        new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                        weighted_single_weights[key] = torch.cat(new, dim=0)
                    else:
                        new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                        weighted_single_weights[key] = torch.cat(new, dim=1)


    torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
    # return model

def Fedilora(selected_clients_set, output_dir,local_dataset_len_dict, epoch, local_ranks):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
            dtype=torch.float32
        ),
    p=1, dim=0)
    print(weights_array)
    weights_by_ranks = {local_ranks[client_id]:{} for client_id in selected_clients_set}
    ranks_sum = {local_ranks[client_id]:0 for client_id in selected_clients_set}
    for k, client_id in enumerate(selected_clients_set):
        local_rank = local_ranks[client_id]
        for rank in weights_by_ranks:
            if rank <= local_rank:
                weights_by_ranks[rank][client_id] = weights_array[k].item()
                ranks_sum[rank] += weights_array[k].item()
    
    for rank in weights_by_ranks:
        for client_id in weights_by_ranks[rank]:
            weights_by_ranks[rank][client_id] /= ranks_sum[rank]
        
    # for k, client_id in enumerate(selected_clients_set):
    #     weights_by_ranks[rank][client_id] /= ranks_sum[rank]

    
    cur_ranks = [cur_rank for cur_rank in weights_by_ranks]
    cur_ranks.sort()
                
    for k, client_id in enumerate(selected_clients_set):
        
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
            "pytorch_model.bin")
        # single_weights is a dict included all lora parameters. key is the parameter name for example layer1.loraA = tensor()
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        #print(single_weights)
        #print("y")
        x = 0    
        max_lora = max(local_ranks)
        my_rank = local_ranks[client_id]
        if k == 0:
            weighted_single_weights = single_weights
            for key in weighted_single_weights.keys():
                if single_weights[key].shape[0] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                    weighted_single_weights[key] = pad(weighted_single_weights[key])
                    last_rank_index = 0
                    for compared_rank in cur_ranks:
                        if my_rank >= compared_rank:
                            weighted_single_weights[key][last_rank_index:compared_rank,] = \
                            weighted_single_weights[key][last_rank_index:compared_rank,] * (weights_by_ranks[compared_rank][client_id])
                            last_rank_index = compared_rank
                    # weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                elif single_weights[key].shape[1] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                    weighted_single_weights[key] = pad(weighted_single_weights[key])
                    last_rank_index = 0
                    for compared_rank in cur_ranks:
                        if my_rank >= compared_rank:
                            weighted_single_weights[key][:,last_rank_index:compared_rank] = \
                            weighted_single_weights[key][:,last_rank_index:compared_rank] * (weights_by_ranks[compared_rank][client_id])
                            last_rank_index = compared_rank
                    # weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
        else:
            for key in single_weights.keys():
                #print(single_weights[key].shape)
                if single_weights[key].shape[0] == local_ranks[client_id]:
                    # pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                    last_rank_index = 0
                    for compared_rank in cur_ranks:
                        if my_rank >= compared_rank:
                            single_weights[key][last_rank_index:compared_rank,] = \
                                single_weights[key][last_rank_index:compared_rank,] * (weights_by_ranks[compared_rank][client_id])
                            # single_weights[key][last_rank_index:compared_rank,] = \
                            # pad(single_weights[key])[last_rank_index:compared_rank,] * (weights_by_ranks[compared_rank][client_id])
                            weighted_single_weights[key][last_rank_index:compared_rank,] += single_weights[key][last_rank_index:compared_rank,]
                            last_rank_index = compared_rank
                    # single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                    # weighted_single_weights[key] += single_weights[key]
                elif single_weights[key].shape[1] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                    last_rank_index = 0
                    for compared_rank in cur_ranks:
                        if my_rank >= compared_rank:
                            single_weights[key][:,last_rank_index:compared_rank] = \
                                single_weights[key][:,last_rank_index:compared_rank] * (weights_by_ranks[compared_rank][client_id])
                            # single_weights[key][:,last_rank_index:compared_rank] = \
                            #     pad(single_weights[key])[:,last_rank_index:compared_rank] * (weights_by_ranks[compared_rank][client_id])
                            weighted_single_weights[key][:,last_rank_index:compared_rank] += single_weights[key][:,last_rank_index:compared_rank]
                            last_rank_index = my_rank
                    # single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                    #print(single_weights[key][255,32])
                    # weighted_single_weights[key] += single_weights[key]
    total_norm = 0.0
    for _, param in weighted_single_weights.items():
        total_norm += torch.norm(param, p=2).item() ** 2  # accumulation square
    print('Aggregating norm2:', total_norm ** 0.5)
    torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))


def HetLora(selected_clients_set, output_dir, epoch, local_ranks):
    # This weights_array is set as key = client_id
    weights_array = compute_F_norm_weight(
        output_dir = output_dir,epoch = epoch,selected_clients_set = selected_clients_set,
    )
    print(weights_array)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
            "pytorch_model.bin")
        # single_weights is a dict included all lora parameters. key is the parameter name for example layer1.loraA = tensor()
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        #print(single_weights)
        #print("y")
        x = 0    
        max_lora = max(local_ranks)
        if k == 0:
            weighted_single_weights = single_weights
            for key in weighted_single_weights.keys():
                if single_weights[key].shape[0] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[client_id])
                elif single_weights[key].shape[1] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                    weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[client_id])
        else:
            for key in single_weights.keys():
                #print(single_weights[key].shape)
                if single_weights[key].shape[0] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                    single_weights[key] = pad(single_weights[key]) * (weights_array[client_id])
                    weighted_single_weights[key] += single_weights[key]
                elif single_weights[key].shape[1] == local_ranks[client_id]:
                    pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                    single_weights[key] = pad(single_weights[key]) * (weights_array[client_id])
                    #print(single_weights[key][255,32])
                    weighted_single_weights[key] += single_weights[key]
    
    torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
    
def Aggregate(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, algo, lora_r, heter, local_ranks):
    if algo.lower() == 'flora':
        Flora(
            model = model,
            selected_clients_set = selected_clients_set,
            output_dir = output_dir,
            local_dataset_len_dict = local_dataset_len_dict,
            epoch = epoch,
            lora_r=lora_r,
            heter = heter,
            local_ranks= local_ranks,
        )
    
    elif algo.lower() == 'hetlora':
        # This is for zero_padding
        HetLora(
            selected_clients_set = selected_clients_set,
            output_dir = output_dir,
            epoch = epoch,
            local_ranks= local_ranks,
        )
    elif algo.lower() == 'fedilora':
        Fedilora(
            selected_clients_set = selected_clients_set,
            output_dir = output_dir,
            local_dataset_len_dict = local_dataset_len_dict,
            epoch = epoch,
            local_ranks= local_ranks,
        )
        # HetLora(
        #     selected_clients_set = selected_clients_set,
        #     output_dir = output_dir,
        #     epoch = epoch,
        #     local_ranks= local_ranks,
        # )