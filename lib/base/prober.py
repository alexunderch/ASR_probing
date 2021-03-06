import torch
from datasets import load_from_disk
from datasets import Dataset, DatasetDict
from .profilers import CheckPointer, ProbingProfiler, MyLogger
from .utils import print_if_debug
from .constants import Constants
from .processing import Processor

import os
import gc
import numpy as np
from collections import Callable
from typing import Dict, Union


class Prober:
    def __init__(self, model_type, model_path: str, writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None, phoneme: bool = False) -> None:
        """ Probing tasks class.
        Args:
            model_type, a HuggingFace model class to probe
            model_path, str: path to model in Hugging Face repo
            data, Dataset: optional, Hugging Face Dataset class 
                           default = None
            device: torch.device, accelerator
            init_strategy: str, flag (use randomly initialized model or downloaded from repo)
                             supported strategies: 
                                -- full 
                                -- (only) encoder
                                -- (only) feature_extractors
                                default = None
            phoneme, bool: whether to do CTC probing or CE probing
                           default = False
        """
        self.cc = Constants
        print_if_debug("downloading staff...", self.cc.DEBUG)
        self._activate_profilers()
        if model_path is not None and model_type is not None: self.model = model_type.from_pretrained(model_path, cache_dir = self.cc.CACHE_DIR).to(device)
        self.writer = writer
        self.data = data
        self.device = device
        self.use_ctc = phoneme
        self.dataprocessor = None

    def _activate_profilers(self):
        self.writer = None
        self.checkpointer = CheckPointer(self.cc.CHECKPOINTING_DIR)
        self.logger = MyLogger(os.path.join(self.cc.CHECKPOINTING_DIR, "logg.log"))
        self.profiler = ProbingProfiler(self.cc.CHECKPOINTING_DIR)
        self.profiler.on() if self.cc.PROFILING else self.profiler.off() 
        self.profiler.profile()

    def _define_dataprocessor(self, proc: Processor) -> None: 
        """Assigning a tokenizer used for CTC-training."""
        self.dataprocessor = proc

    def get_resources(self, load_data: bool = False, data_path: str = None, checkpoint_path: Union[str, Dict] = None, batch_size: int = 100, 
                      poisoning_ratio: float = 0, poisoning_mapping: Callable = None, **kwargs) -> None:
        """
        Args:
          load_data, bool: optional flag, whether load data from external resources or not, ONLY DISK MODE SUPPORTED;
                           default = False
          data_path, str: optional, active only if load_data = True;
                          default = None
          checkpoint_path, str: a path to pretrained model checkpoint or model state dict itself
                                default = None
          batch_size, int: optional;
                           default = 100
          poisoning_ratio, float: the ratio of adding misleading labels to the data (0 -- None, 1 -- fully random);
                           default = 0.       
          poisoning_mapping, callable: the mapping of poisoned labels,
                                       default = None
        """
        print_if_debug("collecting data...", self.cc.DEBUG)
        
        def poison_data(batch, n_classes: int, ratio: float = 0.01, mapping = None):
            """Adding misleading labels"""
            assert ratio > 0. and ratio <= 1.
            if np.random.random() < ratio:
                if mapping is not None: batch['label'] = mapping(batch['label']) 
                else: batch['label'] = np.random.randint(0, n_classes)
            return batch        

        if checkpoint_path is not None:
            if isinstance(checkpoint_path, str) and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location = self.device)
                self.model.load_state_dict(ckpt)
            elif isinstance(checkpoint_path, dict):  self.model.load_state_dict(checkpoint_path)
            else:
                try:
                    print_if_debug("trying to load a huggingface checkpoint...", self.cc.DEBUG)
                    self.model = self.model.from_pretrained(checkpoint_path, cache_dir = self.cc.CACHE_DIR).to(self.device)
                except: print("Nothing has been loaded")
        
        if load_data: 
            assert isinstance(data_path, str)
            self.data =  load_from_disk(data_path, **kwargs)
        
        if poisoning_ratio > 0.: 
            self.data = self.data.map(poison_data, fn_kwargs = {"n_classes": np.max(self.data['label']),
                                                                                    'ratio': poisoning_ratio,
                                                                                     'mapping': poisoning_mapping})

    
        splitted_dataset = self.data.train_test_split(test_size = 0.25, seed = 42)
        if not self.use_ctc:    
            weights = 1. / np.bincount(splitted_dataset['train']['label'])
            self.class_weight = np.array([weights[l] for l in splitted_dataset['train']['label']])
            test_weights = 1. / np.bincount(splitted_dataset['test']['label'])

        self.dataloader = torch.utils.data.DataLoader(splitted_dataset['train'], batch_size = batch_size,
                                                      sampler = torch.utils.data.WeightedRandomSampler(self.class_weight, len(self.class_weight)) if not self.use_ctc else None)
        
        self.validloader = torch.utils.data.DataLoader(splitted_dataset['test'], batch_size = batch_size,
                                                      sampler = torch.utils.data.WeightedRandomSampler(np.array([test_weights[l] for l in splitted_dataset['test']['label']]),
                                                                                                       len(splitted_dataset['test']['label'])) if not self.use_ctc else None)
    def _clear_cache(self):
         if self.device.type == 'cuda':
            with torch.no_grad(): torch.cuda.empty_cache()
            gc.collect()

    def make_probe(self, *args, **kwagrgs) -> dict:
        """ Main method to do a probing task
            Args:
                prober, callable object: funtion with argumnents __data__ and __labels__ to make probing classification (returns metric value of the task)
                enable_grads, bool: optional flag, whether to propagate grads or not
                                    default = False
                use_variational, bool: optional flag, whether to use variational prober or not
                                      default = False
                layers, list: optional list layers indexes to probe (shoud be 0 < layers < #hiddenLayers)
                              default = [0]
                from_memory, str: optionally load probing data from memory (currently deprecated)
                save_outputs, bool: optional flag, whether to save probing data
                                    default = False
                task_tilte, dict: optional way to save probing data, active only if save_outputs = True;
                                default = None
            Returns:
              result, np.ndarray: float array of [#layers, ] output metric results
        """
        raise NotImplementedError("")