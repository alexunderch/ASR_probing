from typing import Dict, Union
from .base.utils import NumpyEncoder, print_if_debug, DatasetSplit
from .base.constants import Constants

from .base.prober import Prober
import torch
from datasets import load_dataset, load_from_disk, Audio
from transformers import logging as log_models
from datasets import Dataset, DatasetDict, set_caching_enabled, logging as log_data

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Callable
class Probing_pipeline:
    def __init__(self, writer: torch.utils.tensorboard.SummaryWriter, device: torch.device,
                 feature: str, model_path: str, data: Dataset = None, lang: str = None, split: DatasetSplit = None) -> None:
        """Hugging Face Dataset wrapper for ASR probing
        Args:
            writer, SummaryWriter: tensorboard writer to debug and visualize all probing process
            feature, str: name of dataset's feature column to make probing onto
            model_path, str: path to model in Hugging Face repo
            custom_feature, bool: optional flag, a possibility to add feature not from original set
            data, Dataset: optional, Hugging Face Dataset class;
                           default = None
            lang, str: an optional feature to this class to determine necessary dataset options
                       default = None
            split, DatasetSplit: a HuggingFace dataset splits (look docs) separated by ';' or "all" to use all available data.

        """
        self.cc = Constants
        log_data.set_verbosity(40 if not self.cc.DEBUG else 20)
        log_models.set_verbosity(40 if not self.cc.DEBUG else 20)

        self.writer = writer
        self.lang = lang
        self.split = split
        self.feature = feature
        self.model_path = model_path
        self.dataset = data
        self.device = device
        self.f_set = None
    

    def load_data(self, from_disk: bool, data_path: str = None, **kwargs) -> None:
        """Custom dataloader
        Args:
            from_disk: bool, flag if load data from disk checkpoint or from the Internet
                             default = False
            data_path, str: optional, active only if from_disk = True;
                            default = None      
        """
        print_if_debug("loading data...", self.cc.DEBUG)

        if from_disk:
            assert isinstance(data_path, str) 
            self.dataset =  load_from_disk(data_path, **kwargs)
            if self.split is not None and isinstance(self.dataset, DatasetDict): self.dataset = self.split.split_str(self.dataset)
        elif data_path is not None: 
            assert isinstance(data_path, str) 
            self.dataset = load_dataset(data_path, name = self.lang, split = None,
                                        **kwargs)
            self.dataset = self.split.split_str(self.dataset)         
        else: assert self.dataset is not None
        # if data_column == 'speech': self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate = 16_000))
        # self.own_feature_set = own_feature_set; self.only_custom_features = only_custom_features

    
    def get_dataset(self): return self.dataset
    def get_feature_set(self): 
        """All labels """
        return self.f_set
    
    def run_probing(self, probing_taskk: Prober, probing_fn, layers: list, checkpoint_path: Union[str, Dict] = None, enable_grads = False, use_variational: bool = False, init_strategy: str = None, plotting_fn: Callable = None, 
                    save_checkpoints: bool = False, plotting_config: dict = None, **kwargs) -> dict:
        """Main probing runner
        Args:
           probing_taskk, Prober: an instance of Prober class (model to be probed)
           probing_fn, init_strategy -- look at Prober docs
           checkpoint_path: str or dict: a path to pretrained model checkpoint or model state dict itself
                                         default = None
           use_variational, bool: optional flag, whether to use variational prober or not
                                  default = False
           enable_grads, bool: optional flag, whether to propagate grads or not
                               default = False
           plotting_fn: callable, optional way to plot results
                       default = None
           save_checkpoints, bool: an optional flag, whether to save checkpoints
                                   defalult = False
           plotting_config, dict ({"title": str, "metrics": list of used in pro bing fn metrics, "save_path": str}), default = None
        """
        probing_task = probing_taskk(self.model_path, self.writer, data = self.dataset, init_strategy = init_strategy, device = self.device)
        probing_task.get_resources(load_data = False, checkpoint_path = checkpoint_path, batch_size = self.cc.BATCH_SIZE, **kwargs)
       

        probing_results = probing_task.make_probe(probing_fn, use_variational = use_variational, enable_grads = enable_grads, layers = layers, 
                                                  save_outputs = save_checkpoints, task_title = plotting_config)
        
        json.dump({"data": probing_results, "config": plotting_config}, 
                  open(os.path.join(plotting_config['save_path'], plotting_config['title'] + ".json"), 'w' ), cls = NumpyEncoder)
        
        if plotting_fn is not None:
            assert isinstance(plotting_fn,  type(lambda x: None))
            assert isinstance(plotting_config, dict)
            plotting_fn(probing_results, plotting_config)
            plt.show()

        return probing_results

    def cleanup(self):
        """Erasing all cache
        """
        if self.dataset.cleanup_cache_files(): return "succeed"
        else: return "nothing has been deleted"
        
    def disable_cache(self): return set_caching_enabled(False)
    def enable_cache(self): return set_caching_enabled(True)

    def __repr__(self):
        return "Used data: {} \n ".format(self.dataset) +\
                "Used feature {} with set of values = {} \n".format(self.feature, self.f_set) +\
                "Used model: {}".format(self.model_path)
