import os
from llava.eval import model_vqa
from llava.eval import (
    eval_textvqa,
    eval_optionvqa
)

class EvalArgs:
    def __init__(self,**kwargs):
        # self.model_path = kwargs.get('model_path',None)
        # self.model_base = kwargs.get('model_base',None)
        self.model= kwargs.get('model',None)
        self.tokenizer= kwargs.get('tokenizer',None)
        self.image_processor= kwargs.get('image_processor',None)
        self.processor = kwargs.get('processor',None)
        
        self.question_file = kwargs.get('question_file',None)
        self.num_chunks, self.chunk_idx = kwargs.get('test_parts',3), kwargs.get('chunk_idx',0)
        self.answers_file = kwargs.get('answers_file',None)
        self.conv_mode = kwargs.get('temperature','vicuna_v1')
        
        self.image_folder = kwargs.get('image_folder',None)
        self.temperature=  kwargs.get('temperature',0.2)
        self.top_p = kwargs.get('top_p',None)
        self.num_beams = kwargs.get('num_beams',1)
        self.mm_use_im_start_end = kwargs.get('mm_use_im_start_end',False)
        
def llava_evaluation(
    epoch, 
    client_id, 
    model,processor, 
    tokenizer, 
    image_processor, 
    data_path,
    image_folder = None, 
    file_format = '', 
    test_parts = 3,
    output_dir = '',
):
    test_data_path = os.path.join(data_path,'test',file_format.format(client_id))
    test_answers_dir = os.path.join(output_dir,'ans',str(client_id))
    os.makedirs(name = test_answers_dir, exist_ok=True)
    test_answers_path = os.path.join(test_answers_dir,str(epoch))
    evalargs = EvalArgs(
        # {
            # "model_path":model_path,
            # "model_base":model_base,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            question_file=test_data_path,
            answers_file=test_answers_path,
            image_folder=image_folder,
            mm_use_im_start_end = False,
            processor = processor,
            test_parts = test_parts,
        # }
    )
    # First generate answers file
    
    
    if data_path.find('scienceqa') >= 0:
        model_vqa.scienceqa_eval_model(evalargs)
    # Then evalute the performance by comparing answers and groud_truth
        samples, acc = eval_optionvqa.eval_single(test_data_path,test_answers_path)
    else:
        model_vqa.eval_model(evalargs)
        samples, acc = eval_textvqa.eval_single(test_data_path,test_answers_path)
    return samples, acc
    
    