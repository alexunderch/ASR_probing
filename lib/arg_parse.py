import argparse
from .base.processing import Processor
import torch
from typing import Union, Dict
def add_nlp_dataset_args(parser: argparse.ArgumentParser):
    """Args for hardcoded NLP-datasets"""
    parser.add_argument('--dataset_type',
                        type = str,
                        help = "the name of a hardcoded dataset. The following ones are available: 'senteval', 'person', 'DiscoEval'('DC', 'SP', 'PDTB') (others are in process)")
    parser.add_argument('--dataset_split',
                        type = str,
                        choices = ('train', 'dev', 'test'),
                        help = "the dataset split which would be used for probing (default = 'test')")
                        
    parser.add_argument('--filepath ',
                        type = str,
                        help = "a filepath to downloaded dataset files (for 'senteval' it should be like filename.txt); Note: for 'DiscoEval' is should be ('DC', 'SP', 'PDTB') -- the name of used set.")

    parser.add_argument('--dataset_name ',
                        type = str,
                        help = "a name the dataset should be saved under")

    parser.add_argument('--feature_column ',
                        type = str,
                        default = 'label',
                        help = "a column which should be probe on (should be in the probing dataset")
    
    parser.add_argument('--data_column  ',
                        type = str,
                        default = 'text', 
                        help = "a column which should be probe on (should be in the probing dataset")
    args = parser.parse_args()
    return args

def add_huggingface_dataset_args(parser: argparse.ArgumentParser):
    """Args for special formatted for HuggingFace lib. datasets"""
    args = parser.parse_args()
    return args

def add_custom_dataset_args(parser: argparse.ArgumentParser):
    """Args for special formatted datasets"""
    args = parser.parse_args()
    return args

def add_experiment_args(parser: argparse.ArgumentParser):
    """Args to set up model and processors"""
    parser.add_argument('--tokenizer',
                        type = Processor,
                        help =   "a tokenizer class")
       
    parser.add_argument('--model_path',
                        type = str,
                        default = None,
                        help = " a name of model to use ('bert', 't5' for nlp tasks); An important note: if dataset_name if not 't5' or 'bert' (then models load automatically), you should set this argument like 't5-small', 'bert-small-cased' etc.")     
   
    parser.add_argument('--enable_grads',
                        type = bool,
                        default = False,
                        help = "a flag whether backprop through model layers or not")
    parser.add_argument('--save_checkpoints',
                        type = bool,
                        default = False,
                        help = "save checkpoints for each layer or not ")            
    args = parser.parse_args()
    return args

def add_model_args(parser: argparse.ArgumentParser):
    """Args to set up model options"""
    parser.add_argument('--checkpoint_path',
                        type = Union[str, Dict],
                        default = None,
                        help = "a path to checkpoint for standard torch.nn.Module models or state dict for HuggingFaceModels ")
    parser.add_argument('--model_init_strategies ',
                        type = list,
                        default = [None, "full"],
                        help = "a list of model init. stategies ")
    parser.add_argument('--probing_fn',
                        type = torch.nn.Module,
                        help = "a class instance of probing classier ")

    parser.add_argument('--use_mdl',
                        type = bool,
                        default = False,
                        help = "use MDL (variational approach) or not.")
    
    args = parser.parse_args()
    return args

def add_auxillary_args(parser: argparse.ArgumentParser):
    """Args-helpers"""
    parser.add_argument('--device',
                        type = torch.device,
                        default = torch.device('cpu'),
                        help = "a probing device")
    parser.add_argument('--download_data',
                        type = bool,
                        default = True,
                        help = "whether to use prober the first time or not")
    args = parser.parse_args()
    return args

def get_nlp_args():
    """Get arguments needed in .py."""
    parser = argparse.ArgumentParser('Running NLP Probing.')
    add_nlp_dataset_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    add_auxillary_args()
    args = parser.parse_args()
    return args

def get_auxillary_args():
    """Get arguments needed in .py."""
    parser = argparse.ArgumentParser('Running NLP Probing.')
    add_huggingface_dataset_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    add_auxillary_args()
    args = parser.parse_args()
    return args

def get_custom_args():
    """Get arguments needed in .py."""
    parser = argparse.ArgumentParser('Running NLP Probing.')
    add_huggingface_dataset_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    add_auxillary_args()
    args = parser.parse_args()
    return args
