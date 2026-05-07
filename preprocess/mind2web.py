from datasets import load_dataset
from tqdm import tqdm
import json

import random
import os
import json


def preprocess(dataset, image_dir, json_dir, clients_num, train_test_ratio = 0.85):
    clients_train_list = [[] for i in range(clients_num)]
    clients_test_list = [[] for i in range(clients_num)]
    train_counter = [0] * 10
    test_counter = [0] * 10
    train_test_ratio = int(train_test_ratio * 100)
    os.makedirs(image_dir,exist_ok=True)
    for item in tqdm(dataset['train']):
        id = item['id']
        image = item['image'].convert('RGB')
        image_path = os.path.join(image_dir,f'{id}.jpg')
        image.save(image_path)
        conversations = item['conversations']
        ranidx = random.randint(0,clients_num-1)
        
        # distribute to train or test
        if random.randint(1,100) <= train_test_ratio:
            clients_train_list[ranidx].append(
                {
                    "sid":train_counter[ranidx],
                    "id":id,
                    "image": image_path,
                    "conversations": conversations,
                }
            )
            train_counter[ranidx] += 1
        else:
            clients_test_list[ranidx].append(
                {
                    "sid":test_counter[ranidx],
                    "id":id,
                    "image": image_path,
                    "conversations": conversations,
                }
            )
            test_counter[ranidx] += 1

    train_prefix = os.path.join(json_dir, str(clients_num), 'train')
    test_prefix = os.path.join(json_dir, str(clients_num), 'test')
    os.makedirs(train_prefix,mode = 766, exist_ok=True)
    os.makedirs(test_prefix,mode = 766, exist_ok=True)
    
    for idx, client_data in enumerate(tqdm(clients_train_list)):
        client_jsonfile = os.path.join(train_prefix, f'{idx}.json')
        with open(client_jsonfile,'w') as cf:
            json.dump(obj = client_data, fp = cf, indent = 4)
    
    
    for idx, client_data in enumerate(tqdm(clients_test_list)):
        client_jsonfile = os.path.join(test_prefix,  f'recap_{idx}.json')
        with open(client_jsonfile,'w') as cf:
            json.dump(obj = client_data, fp = cf, indent = 4)

if __name__ == '__main__':
    ds = load_dataset("neulab/Mind2Web_train_llava")
    print(ds.keys())
    # for key in ds:
    #     print(ds[key].keys())
    print(ds['train']['conversations'][1])
    image_dir = '/home/lishan/data/Mind2Web/images'
    json_dir = '/home/lishan/workspace/2025EMNLP-FLME/Mind2Web'
    clients_num = 10

    preprocess(ds,image_dir,json_dir,clients_num)