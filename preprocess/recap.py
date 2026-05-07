from datasets import load_dataset
from tqdm import tqdm
import json

import random
import os
import json
import setseeds

def preprocess(dataset, image_dir, json_dir, clients_num, train_test_ratio = 0.9):
    clients_train_list = [[] for i in range(clients_num)]
    # clients_test_list = [[] for i in range(clients_num)]
    test_list = []
    train_counter = [0] * 10
    test_counter = 0
    train_test_ratio = int(train_test_ratio * 100)
    os.makedirs(image_dir,exist_ok=True)
    for item in tqdm(dataset['train']):
        id = item['id']
        image_path = os.path.join(image_dir,f'{id}.jpg')
        if not os.path.exists(image_path):
            image = item['image'].convert('RGB')
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
            test_list.append(
                {
                    "sid":test_counter,
                    "id":id,
                    "image": image_path,
                    "conversations": conversations,
                }
            )
            test_counter += 1
            

    train_prefix = os.path.join(json_dir, str(clients_num), 'train')
    test_prefix = os.path.join(json_dir, str(clients_num), 'test')
    os.makedirs(train_prefix, exist_ok=True)
    os.makedirs(test_prefix, exist_ok=True)
    
    for idx, client_data in enumerate(tqdm(clients_train_list)):
        client_jsonfile = os.path.join(train_prefix, f'{idx}.json')
        with open(client_jsonfile,'w') as cf:
            json.dump(obj = client_data, fp = cf, indent = 4)
    
    client_jsonfile = os.path.join(test_prefix,  f'recap_test.json')
    with open(client_jsonfile,'w') as cf:
        json.dump(obj = test_list, fp = cf, indent = 4)


if __name__ == '__main__':
    ds = load_dataset("lmms-lab/LLaVA-ReCap-118K")
    setseeds.set_random_seed(666)
    print(ds['train']['conversations'][0])
    image_dir = '/hpcfs/users/a1905721/ReCap-118K/images'
    json_dir = '/hpcfs/users/a1905721/workspace/2025EMNLP-FLME/data_recaps'
    clients_num = 10

    preprocess(ds,image_dir,json_dir,clients_num)