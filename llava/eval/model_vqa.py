import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import inspect

import random
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_random_test_set(questions,num_chunks):
    test_size = len(questions)
    num_each_chunk = int(test_size / num_chunks)
    return random.sample(questions, num_each_chunk)

def eval_model(args):
    # Model
    '''
    ## args explainations:
    **model_path**: evaluation lora model config path\\
    **model_base**: base_model for example llava\\
    **question_file**: a json file each line is a question\\
    **num_chunks, chunk_idx**: partitioning the question into num_chunks, select which chunk to evaluate\\
    **answers_file**: prediction file\\
    **conv_mode**: default or conv_vicuna_v0 or conv_vicuna_v1\\
    **image_folder**: image_foler\\
    **temperature**: default temperature = 0.2\\
    **top_p**: None\\
    **num_beams**: default 1
    '''
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model,tokenizer,image_processor,max_token_size = args.model,args.tokenizer,args.image_processor,1024
    processor = args.processor
    with open(args.question_file, "r") as qf:
        questions = json.load(qf)
    # questions = [json.loads(q) for q in open(args.question_file, "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = get_random_test_set(questions,args.num_chunks)
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers = []
    for line in tqdm(questions):
        idx = line["sid"] # sid
        image_file = line["image"] #image path
        # qs = line["text"] # 
        qs = line['conversations'][0]['value']
        cur_prompt = qs
        if args.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        elif qs == '<image>':
            qs = 'Please describe it.\n ' + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        
        image = Image.open(image_file).convert('RGB')
        # image_tensor = process_images([image], image_processor, model.config)[0]
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # print(inspect.signature(model.generate))
        device = model.device
        with torch.inference_mode():
            # output_ids = model(
            #     input_ids,
            #     pixel_values=image_tensor,
            #     # max_new_tokens=max_token_size
            # )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            # output_ids = model.generate(
            #     input_ids,
            #     pixel_values=image_tensor.cuda(),
            #     max_new_tokens=max_token_size
            # )
            # output_ids = model.base_model.generate(
            #     input_ids,
            #     # pixel_values=image_tensor.unsqueeze(0).half().cuda(),
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=max_token_size,
            #     use_cache=True
            # )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        answers.append(
            {
                "sid": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
            }
        )
        # ans_file.write(json.dumps() + "\n")
    # ans_file.flush()
    ans_file = open(answers_file, "w")
    json.dump(answers,ans_file)
    ans_file.close()

def scienceqa_eval_model(args):
    # Model
    '''
    ## args explainations:
    **model_path**: evaluation lora model config path\\
    **model_base**: base_model for example llava\\
    **question_file**: a json file each line is a question\\
    **num_chunks, chunk_idx**: partitioning the question into num_chunks, select which chunk to evaluate\\
    **answers_file**: prediction file\\
    **conv_mode**: default or conv_vicuna_v0 or conv_vicuna_v1\\
    **image_folder**: image_foler\\
    **temperature**: default temperature = 0.2\\
    **top_p**: None\\
    **num_beams**: default 1
    '''
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model,tokenizer,image_processor,max_token_size = args.model,args.tokenizer,args.image_processor,1024
    processor = args.processor
    with open(args.question_file, "r") as qf:
        questions = json.load(qf)
    # questions = [json.loads(q) for q in open(args.question_file, "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = get_random_test_set(questions,args.num_chunks)
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers = []
    for line in tqdm(questions):
        idx = line["sid"] # sid
        image_file = line["image"] #image path
        # qs = line["text"] # 
        qs = line['conversations'][0]['value']
        cur_prompt = qs
        if args.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        elif qs == '<image>':
            qs = 'Please describe it.\n ' + qs
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = qs + ' ASSISTANT: The answer is'
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        
        image = Image.open(image_file).convert('RGB')
        # image_tensor = process_images([image], image_processor, model.config)[0]
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # print(inspect.signature(model.generate))
        device = model.device
        with torch.inference_mode():
            # output_ids = model(
            #     input_ids,
            #     pixel_values=image_tensor,
            #     # max_new_tokens=max_token_size
            # )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            # output_ids = model.generate(
            #     input_ids,
            #     pixel_values=image_tensor.cuda(),
            #     max_new_tokens=max_token_size
            # )
            # output_ids = model.base_model.generate(
            #     input_ids,
            #     # pixel_values=image_tensor.unsqueeze(0).half().cuda(),
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=max_token_size,
            #     use_cache=True
            # )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # outputs = outputs[len(prompt):]
        output_idx = outputs.find('The answer is')
        if output_idx >= 0:
            outputs = outputs[output_idx:]
        ans_id = shortuuid.uuid()
        answers.append(
            {
                "sid": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
            }
        )
        # ans_file.write(json.dumps() + "\n")
    # ans_file.flush()
    ans_file = open(answers_file, "w")
    json.dump(answers,ans_file)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
