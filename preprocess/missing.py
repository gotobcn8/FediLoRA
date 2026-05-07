
import os
import json
import random
import setseeds
MISSING = 'missing'

def gen_missing_indexs(json_dir, num_clients, missing_rate, file_format):
    missing_dir = os.path.join(json_dir,str(num_clients),MISSING,str(missing_rate))
    json_dir = os.path.join(json_dir,str(num_clients),'train')
    os.makedirs(missing_dir, exist_ok=True)
    for cid in range(num_clients):
        ''' 
        Because I pre-set data directory as data_recaps/10/train/recap_{id}.json, 
        Exactly no need to set recap_ as a prefix for other datasets 
        '''
        client_json_file = os.path.join(json_dir, file_format + str(cid) + '.json')
        # Open the original data file
        with open(client_json_file,'r') as f:
            data = json.load(f)
        # Get data size
        data_size = len(data)
        missing_nums = int(missing_rate * data_size)
        indexs = random.sample(range(data_size),missing_nums)
        missing_indexs = {}
        for index in indexs:
            missing_indexs[index] = random.randint(0,1)
        #  Dump the missing files
        with open(os.path.join(missing_dir, str(cid) + '.json'),'w') as p:
            json.dump(missing_indexs,p)
    
if __name__ == '__main__':
    # Please change this directory to your data directory (data_recaps is the data base directory)
    setseeds.set_random_seed(666)
    json_dir = '/home/lishan/workspace/2025EMNLP-FLME/scienceqa'
    file_format = '' # recaps_
    num_clients = 10

    gen_missing_indexs(json_dir = json_dir,num_clients = num_clients,missing_rate = 0.6, file_format = file_format)
    # preprocess(ds,image_dir,json_dir,clients_num)