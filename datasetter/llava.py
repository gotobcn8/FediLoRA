import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import pickle
import random

NONE_TOKEN = 5642
USER_TOKEN = 'USER:'
ASSISTANT_TOKEN = 'ASSISTANT:'

class LlavaStyleDataset(Dataset):
    def __init__(self, json_path, processor = None,max_length=1024, missing_indexs = None,data_subset_rate = 1.0):
        with open(json_path,'r') as f:
            self.data = json.load(f)
        # self.samples = [json.loads(line) for line in open(json_path)]
        # self.tokenizer = tokenizer
        # self.image_processor = image_processor
        self.max_length = max_length
        self.processor = processor
        self.missing_indexs = missing_indexs
        self.data_subset_rate = data_subset_rate
        self.set_subset()
        
    def set_subset(self):
        if self.data_subset_rate < 1.0:
            self.data = random.sample(self.data, int(len(self.data) * self.data_subset_rate))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        # .convert("RGB")
        conv = item["conversations"]
        sid = item["sid"]
        if str(sid) in self.missing_indexs:
            if self.missing_indexs[str(sid)] == 1:
                conv[1]['value'] = "None None None"
        new_conv = []
        raw_prompt = ''
        for i,words in enumerate(conv):
            if words['from'] == 'human':
                dialog = {
                    "role":"user",
                    "content":[
                        {"type":'<image>'},
                        # {"type":"text","text":"What's shown in this image?"}
                    ]
                }
                raw_prompt = words['value']
            else:
                dialog = {
                    "role":"assistant",
                    "content":[
                        {"type":"text", 'text':words['value']},
                    ]
                }
            new_conv.append(dialog)
        text_prompt = self.processor.apply_chat_template(
            new_conv,
            add_generation_prompt=True
        )
        
        # prompt especially for next
        if raw_prompt == '<image>':
            raw_prompt = 'Please describe it.\n ' + raw_prompt
        
        if text_prompt.startswith(USER_TOKEN):
            # text_prompt = starttokens + 'Please describe this image \n <image>' + text_prompt[len(starttokens):]
            text_prompt = USER_TOKEN + raw_prompt + text_prompt[len(USER_TOKEN):]
        if text_prompt.endswith(ASSISTANT_TOKEN):
            text_prompt = text_prompt[:len(text_prompt) - len(ASSISTANT_TOKEN)]
        return text_prompt, image, sid

    # def set_missing_values(self):
        

class TrainLLavaModelCollator:
    def __init__(self,processor, missing_indexs):
        # tokenizer, image_processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.processor = processor
        self.padding = True
        self.missing_indexs = missing_indexs
        
    def set_missing(self,image,tokens):
        if image is not None:
            return torch.zeros(image.size())
        else:
            tokens[15:min(35,len(tokens))] = 0
            return tokens
        
    def __call__(self, features:list[dict]) -> dict:
        # print(features[0].keys)
        # pixel_values = [f.pop("pixel_values") for f in features]
        images = [f[1] for f in features]
        prompt_texts = [f[0] for f in features]
        sids = [f[2] for f in features]
        batch = self.processor(text=prompt_texts, images=images, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        
        for i,sid in enumerate(sids):
            if str(sid) in self.missing_indexs:
                if self.missing_indexs[str(sid)] == 0:
                    batch['pixel_values'][i] = torch.zeros(batch['pixel_values'][i].size())
        # missing_idx = random.sample(range(len(features)),1)
        # missing_idx = missing_idx[0]
        # if random.randint(0,1) == 0:
        #     batch["pixel_values"][missing_idx] = self.set_missing(batch["pixel_values"][missing_idx],None)
        # else:
        #     batch["input_ids"][missing_idx] = self.set_missing(None,batch["input_ids"][missing_idx])
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # batch.pop('image_sizes')
        # return {
        #     "input_ids": batch['input_ids'],
        #     "labels": batch['labels'],
        #     "pixel_values": batch['pixel_values'],
        #     "attention_mask": batch['attention_mask'],
        # }
        return batch

# 151646
def check_image(batch,image_token_idx = 151646):
    input_ids = batch['input_ids']
    exists = (input_ids == image_token_idx).any()
    count = (input_ids == image_token_idx).sum()
    print(f"Exists: {exists.item()}, Count: {count.item()}")
    return exists.item(), count.item()