from datasets import load_dataset
from tqdm import tqdm
import json

import random
import os
import json
import setseeds

def preprocess(dataset, image_dir, json_dir, clients_num, train_test_ratio = 0.85):
    clients_train_list = [[] for i in range(clients_num)]
    # clients_test_list = [[] for i in range(clients_num)]
    train_counter = [0] * 10
    test_counter = 0
    train_image_dir = os.path.join(image_dir,'train')
    test_image_dir = os.path.join(image_dir,'test')
    os.makedirs(train_image_dir,exist_ok=True)
    os.makedirs(test_image_dir,exist_ok=True)
    train_prefix = os.path.join(json_dir, str(clients_num), 'train')
    test_prefix = os.path.join(json_dir, str(clients_num), 'test')
    os.makedirs(train_prefix, exist_ok=True)
    os.makedirs(test_prefix, exist_ok=True)
    
    
    test_set = []
    for idx,item in tqdm(enumerate(dataset['test'])):
        image_path = os.path.join(test_image_dir,f'{idx}.jpg')
        if not os.path.exists(image_path):
            image = item['image'].convert('RGB')
            image.save(image_path)
        caption = item['caption']
        
        # distribute to train or test

        test_set.append(
            {
                "sid": test_counter,
                "id": idx,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>"
                    },
                    {
                        "from": "gpt",
                        "value": caption,
                    }
                ]
            }
        )
        test_counter += 1
    
    test_set_jsonfile = os.path.join(test_prefix,  f'sam_test.json')
    with open(test_set_jsonfile,'w') as cf:
        json.dump(obj = test_set, fp = cf, indent = 4)
    
    
    for idx,item in tqdm(enumerate(dataset['train'])):
        
        image_path = os.path.join(train_image_dir,f'{idx}.jpg')
        if not os.path.exists(image_path):
            image = item['image'].convert('RGB')
            image.save(image_path)
        caption = item['caption']
        ranidx = random.randint(0,clients_num-1)
        
        # distribute to train or test

        clients_train_list[ranidx].append(
            {
                "sid": train_counter[ranidx],
                "id": idx,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>"
                    },
                    {
                        "from": "gpt",
                        "value": caption,
                    }
                ]
            }
        )
        train_counter[ranidx] += 1
    
    for idx, client_data in enumerate(tqdm(clients_train_list)):
        client_jsonfile = os.path.join(train_prefix, f'{idx}.json')
        with open(client_jsonfile,'w') as cf:
            json.dump(obj = client_data, fp = cf, indent = 4)
    
    
    
if __name__ == '__main__':
    ds = load_dataset("unography/SAM-LLAVA-20k")
    # for key in ds:
    #     print(ds[key].keys())
    # print(ds['train']['conversations'][1])
    setseeds.set_random_seed(666)
    image_dir = '/home/lishan/data/sam/images'
    json_dir = '/home/lishan/workspace/2025EMNLP-FLME/sam'
    clients_num = 10

    preprocess(ds,image_dir,json_dir,clients_num)
