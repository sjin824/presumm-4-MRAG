# -*- coding: utf-8 -*-
# @Time: 02/12/2024 22:10pm NZDT (UTC+13)
# @Author (not origin): Siyu Jin
# @File: presumm_interface
# @Annotation:  
#   Is the modified version of 'PreSumm/src/train.py'.
#   Iteract with 'PreSumm' and 'PresummSentenceRanker' to rank sentences.
#
#   Original train.py invokes the wrong functions at line: 156, It should invoke 'test_text_ext()' instead of 'test_text_abs()';
#   However, the 'test_text_abs' function is not defined in the original 'PreSumm/src/train_extractive.py'.
#   Therefore, modifications on the 'train.py' and 'train_extractive.py' are required.
#   Modified places are annotated.
#
#   The most important structure variation of invoking:
#                - presumm_train_abstractive_v2.test_abs()
#               |
#   - ranking_request
#               |
#                - presumm_train_exstractive_v2.test_ext()

import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs # Removed 'test_text_abs' from the original code.
from train_extractive import train_ext, validate_ext, test_ext


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ranking_request(configs=False, sample=None, target=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    
    # Zhenyun added a new option 'test_text', Siyu inherited it, then invoked 'test_abs' and 'test_ext' in his way in a branch below. 
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'test_text']) 
    parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    
    # There were 2 new parameters "-text_src" and "-text_tgt" in Zhenyun's code. Invoked by Zhenyun's data_loader.load_text
    # Siyu deleted both while they are not used anymore. In stream rather than whole dataset.
    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-max_ndocs_in_batch", default=6, type=int) # Zhenyun's implementation. Not used in Siyu's but remains.

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    
    # Siyu inherited the following lines from Zhenyun's code, partially.
    if configs:
        args.task = configs['task']
        args.mode = configs['mode']
        args.test_from = configs['test_from']
        args.result_path = configs['result_path']
        args.alpha = configs['alpha']
        args.log_file = configs['log_file']
        args.visible_gpus = configs['visible_gpus']

    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
            
        # This branch is modified by Siyu, different from orginal 'PreSumm/src/models/train.py'.
        # In Zhenyun's modification, there used test_text_abs()' function.
        # Actually in 'PreSumm/src/models/train.py', the functions 'test_text_abs()' and 'test_abs()' are the same.
        # Therefore, we could just invoke 'test_abs()' here.
        elif (args.mode == 'test_text'): 
            test_abs(args=args, text=sample, target=target, device_id=device_id, pt='', step=-1)
            
    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
            
        # This branch is modified by Siyu, different from orginal 'PreSumm/src/models/train.py'.
        # In 'PreSumm/src/models/train.py', there is no such 'test_text_ext()' function (This function was added in Zhenyun's implementation).
        # Given that the functions 'test_text_abs()' and 'test_abs()' are the same,
        # We could just invoke'test_ext()' here.
        elif (args.mode == 'test_text'): 
            # 保留了和zhenyun一样的return
            return test_ext(args=args, text=sample, target=target, device_id=device_id, pt='', step=-1) 