from datasets import load_dataset
from tqdm import tqdm
import json
import uuid
import shutil

import random
import os
import setseeds

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class HuggingFaceDataset:
    def __init__(self, dataset_name, images_path, recaps_path, clients_num=10, train_test_ratio=0.85):
        self.ds = load_dataset(dataset_name, split="train")
        self.clients_num = clients_num
        self.train_test_ratio = train_test_ratio
        self.images_path = images_path
        self.recaps_path = recaps_path
        pass
    
    def convert_json(self, divide_test_to_client=False, mode=0o777, file_format='recap_', mini=True, dataset_name='next'):
        train_path = os.path.join(self.recaps_path, f'{self.clients_num}', 'train')
        test_path = os.path.join(self.recaps_path, f'{self.clients_num}', 'test')
        os.makedirs(train_path, mode=mode, exist_ok=True)
        os.makedirs(test_path, mode=mode, exist_ok=True)
        clients_train_list = [[] for i in range(self.clients_num)]
        clients_test_list = [[] for i in range(self.clients_num)]
        test_list = []
        train_test_split = self.ds.train_test_split(test_size= 1 - self.train_test_ratio)
        train_set = train_test_split['train']
        test_set = train_test_split['test']
        # Process train set
        print("Processing train set...")
        for idx, item in enumerate(tqdm(train_set)):
            if mini and idx == len(train_set) // 100:
                break
            if item['image'] is None:
                continue
            random_client_index = random.randint(0, self.clients_num-1)
            sid = len(clients_train_list[random_client_index])
            new_sample = self.format_obj(item=item, sid=sid)
            self.save_image(new_sample['id'], item['image'])
            clients_train_list[random_client_index].append(new_sample)
        # Process test set
        print("Processing test set...")
        for idx, item in enumerate(tqdm(test_set)):
            if mini and idx == len(test_set) // 100:
                break
            if item['image'] is None:
                continue
            if divide_test_to_client:
                random_client_index = random.randint(0, self.clients_num-1)
                sid = len(clients_test_list[random_client_index])
                new_sample = self.format_obj(item=item, sid=sid)
                self.save_image(new_sample['id'], item['image'])
                clients_test_list[random_client_index].append(new_sample)
            else:
                sid = len(test_list)
                new_sample = self.format_obj(item=item, sid=sid)
                self.save_image(new_sample['id'], item['image'])
                test_list.append(new_sample)
                
        # Save train set to json
        print("Saving train set to JSON...")
        for idx, train_data in enumerate(tqdm(clients_train_list)):
            train_jsonfile = os.path.join(train_path, f"{file_format}{idx}{'_mini' if mini else ''}.json")
            with open(train_jsonfile,'w') as file:
                json.dump(train_data, file, indent = 4)
        
        # Save test set to json
        print("Saving test set to JSON...")
        if divide_test_to_client:
            for idx, client_data in enumerate(tqdm(clients_test_list)):
                client_jsonfile = os.path.join(test_path,  f"{file_format}{dataset_name}_test{'_mini' if mini else ''}.json")
                with open(client_jsonfile,'w') as file:
                    json.dump(client_data, file, indent = 4)
        else:
            client_jsonfile = os.path.join(test_path,  f"{file_format}{dataset_name}_test{'_mini' if mini else ''}.json")
            with open(client_jsonfile,'w') as file:
                json.dump(test_list, file, indent = 4)
    def format_obj(self):
        pass

    def save_image(self, id, new_image, mini=False):
        image_path = os.path.join(self.images_path,f"{id}.jpg")
        if os.path.exists(image_path):
            return
        image = new_image.convert('RGB')
        image.save(image_path)

    def count_items(self):
        images = os.listdir(self.images_path)
        num_train = 0
        num_test = 0
        # Count items
        train_path = os.path.join(self.recaps_path, f"{self.clients_num}", "train")
        test_path = os.path.join(self.recaps_path, f"{self.clients_num}", "test")
        for filename in os.listdir(train_path):
            with open(os.path.join(train_path, filename), 'r') as file:
                data = json.load(file)
                num_train += len(data)
        for filename in os.listdir(test_path):
            with open(os.path.join(test_path, filename), 'r') as file:
                data = json.load(file)
                num_test += len(data)
        print(f"Number of images: {len(images)}")
        print(f"Number of train: {num_train}")
        print(f"Number of test: {num_test}")
    

class ScienceQADataset(HuggingFaceDataset):
    def __init__(self, images_path, recaps_path, clients_num=10, train_test_ratio=0.85):
        super().__init__(
            dataset_name="cnut1648/ScienceQA-LLAVA", 
            images_path=images_path,
            recaps_path=recaps_path,
            clients_num=clients_num,
            train_test_ratio=train_test_ratio
        )
        
    def format_obj(self, item, sid):
        new_id = str(uuid.uuid4())
        new_sample = {
            'sid': sid,
            'id': new_id,
            'image': os.path.join(self.images_path,f"{new_id}.jpg"),
            'conversations': item['conversations']
        }
        return new_sample

if __name__ == "__main__":
    # -------------------------parameters to change---------------------------
    home_dir = os.environ['HOME']
    cur_dir = os.getcwd()
    images_dir = os.path.join(home_dir,'data/scienceqa/images')
    science_path = os.path.join(cur_dir,'scienceqa')
    setseeds.set_random_seed(666)
    num_clients = 10
    train_test_ratio = 0.9
    mode = 0o777
    file_format = ''
    dataset_name = 'scienceqa'
    mini = False
    # ------------------------------------------------------------------------
    os.makedirs(images_dir, mode=mode, exist_ok=True)
    os.makedirs(science_path, mode=mode, exist_ok=True)
    new_dataset = ScienceQADataset(
        images_path=images_dir,
        recaps_path=science_path,
        clients_num=num_clients,
        train_test_ratio=train_test_ratio
    )
    new_dataset.convert_json(mode=mode, file_format=file_format, mini=mini, dataset_name=dataset_name)
    new_dataset.count_items()