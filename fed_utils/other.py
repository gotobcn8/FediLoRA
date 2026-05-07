import torch

NORMAL_ALGO = ['hetlora','fedilora']

def other_function():

    return print("design the other functions you need")



def compute_model_l2_norm(model):
    total_norm = 0.0
    for key, param in model.state_dict().items():
        total_norm += torch.norm(param, p=2).item() ** 2  # accumulation square
    return total_norm ** 0.5