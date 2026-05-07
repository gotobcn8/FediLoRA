import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator, TextVQAGenerationEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    '''
    annotation is groud-truth
    resultfile is output file
    '''
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    # annotations = json.load(open(annotation_file))['data']
    annotations = json.load(open(annotation_file))
    # annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    # results = [json.loads(line) for line in open(result_file)]
    with open(result_file,'r') as rf:
        results = json.load(rf)
    pred_list = []
    gt_answers = []
    for result in results:
        # annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        annotation = annotations[result['sid']]
        # pred_list.append({
        #     "pred_answer": result['text'], # generation predictions
        #     "gt_answers": annotation['conversations'][1]['value'], # ground-truth
        # })
        pred_list.append(result['text'])
        gt_answers.append(annotation['conversations'][1]['value'])
        
    # evaluator = TextVQAAccuracyEvaluator()
    evaluator = TextVQAGenerationEvaluator()
    eval_res = evaluator.eval_pred_list(pred_list, gt_answers)
    print('Samples: {}\nPerformance: {}%\n'.format(len(pred_list), eval_res))
    return len(pred_list) , eval_res

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
