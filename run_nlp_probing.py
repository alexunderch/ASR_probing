from lib.linguistic_utils import NLPDatasetProcessor, make_nlp_dataset
from lib.tokenizers import T5Processor, BertProcessor
from lib.probers import BertOProber, T5Prober
from lib.pipeline import Probing_pipeline, _make_directory_structure
from IPython.display import clear_output
from lib.base.clf import ProberModel
from lib.base.constants import Constants
from collections import Callable
import os
import torch
from torch.utils import tensorboard


def main(config: dict):
    _ = make_nlp_dataset(**config)
    

    

if __name__ == "__main__": main()