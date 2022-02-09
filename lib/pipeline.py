from func_utils import NumpyEncoder, print_if_debug
from probers import Prober
import torch
from datasets import load_dataset, load_from_disk
from transformers import Wav2Vec2Processor, logging as log_models
from datasets import Dataset, DatasetDict, set_caching_enabled, logging as log_data

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
from collections import Callable


class Probing_pipeline:
    def __init__(self, writer: torch.utils.tensorboard.SummaryWriter,
                 feature: str, model_path: str, data: Dataset = None, lang: str = None, split: str = None) -> None:
        """Hugging Face Dataset wrapper for ASR probing
        Args:
            writer, SummaryWriter: tensorboard writer to debug and visualize all probing process
            feature, str: name of dataset's feature column to make probing onto
            model_path, str: path to model in Hugging Face repo
            custom_feature, bool: optional flag, a possibility to add feature not from original set
            data, Dataset: optional, Hugging Face Dataset class;
                           default = None
            (lang, split), str: optional features to this class to determine necessary dataset options
                                default = None

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

    def load_data(self, from_disk: bool, data_path: str = None, own_feature_set: dict = None, only_custom_features: bool = True, **kwargs) -> None:
        """Custom dataloader
        Args:
            from_disk: bool, flag if load data from disk checkpoint or from the Internet
                             default = False
            data_path, str: optional, active only if from_disk = True;
                            default = None      
            own_feature_set, dict (format {feat: int}): optional own mapping to probing labels
                                                    default = None
            only_custom_features, bool:optional flag whether to use only custom features or add "other" ground class,
                                      active only with own_feature_set, default = True
        """
        print_if_debug("loading data...", self.cc.DEBUG)
        if from_disk:
            assert isinstance(data_path, str) 
            self.dataset =  load_from_disk(data_path, **kwargs)
            if self.split is not None and isinstance(self.dataset, DatasetDict): self.dataset = self.dataset[self.split] 
        elif data_path is not None: 
            assert isinstance(data_path, str) 
            self.dataset = load_dataset(data_path, name = self.lang, split = self.split,
                                        **kwargs)         
        else: assert self.dataset is not None
        self.own_feature_set = own_feature_set; self.only_custom_features = only_custom_features

    def _filter_data(self, own_feature_set: dict, only_custom_features: bool) -> None:
        """
        Args: 
          own_feature_set, dict (format {feat: int}): own mapping to probing labels
                                                    default = None
          only_custom_features, bool: flag whether to use only custom features or add "other" ground class,
                                      active only with own_feature_set, default = True

        """
        self.dataset = self.dataset.filter(lambda example: len(example[self.feature].strip()) > 0)
        if own_feature_set is None: self.f_set = {v: k for k, v in enumerate(list(set(self.dataset[self.feature])))}
        else: 
            assert isinstance(own_feature_set, dict)
            self.f_set = own_feature_set            
            if not only_custom_features:
                self.f_set["other"] = np.max(list(self.f_set.values())) + 1
                def foo(batch):
                    if batch[self.feature] not in self.f_set.keys(): batch[self.feature] = "other"
                    return batch                    
                self.dataset = self.dataset.map(foo)
            else: self.dataset = self.dataset.filter(lambda example: example[self.feature] in self.f_set)


    def get_dataset(self): return self.dataset
    def get_feature_set(self): 
        """All labels """
        return self.f_set

    def preprocess_data(self, preprocessinf_fn: Callable, save_path: str = None, drop_columns: list = None, target_processing: Callable = None):
        """
        Args:
            preprocessinf_fn, callable object: prerpocessing dataset function to load all audio, should return the same self.dataset
                                               but with 'speech', 'len_speech', 'sampling_rate' columns
            save_path, str: optional path to save preprocessed data
                            default = None
            drop_columns, list: optional list of string-like columns to drop from the dataset
                                default = None
            target_processing, callable object: prerpocessing dataset function to speech transcript. Shouldn't change the dataset structure
                                                default = None
        """
        print_if_debug("downloading necessary staff...", self.cc.DEBUG)
        processor =  Wav2Vec2Processor.from_pretrained(self.model_path, cache_dir = self.cc.CACHE_DIR)
        def encode_labels(example, feature_column: str):
            """Label Encoder
            """
            example["label"] = self.f_set[example[feature_column]]
            return example

        def add_new_features(batch, max_len: int):
            """Preprocessing audio features with padding to maximum lenght"""
            inputs = processor(batch["speech"], sampling_rate = batch["sampling_rate"], return_tensors = "pt", 
                               padding = 'max_length', truncation = 'max_length', max_length = max_len)
            batch['input_values'] = inputs.input_values
            batch['attention_mask'] = inputs.attention_mask
            return batch

        print_if_debug('reading files...', self.cc.DEBUG)
        if preprocessinf_fn is not None:
            self.dataset = self.dataset.map(preprocessinf_fn, fn_kwargs = {'feature_column': self.feature}, disable_nullable = False)
        print_if_debug('encoding features...', self.cc.DEBUG)
        self._filter_data(self.own_feature_set, self.only_custom_features)
        self.dataset = self.dataset.map(encode_labels, fn_kwargs = {'feature_column': self.feature})

        print_if_debug('processing features...', self.cc.DEBUG)
        self.dataset = self.dataset.map(add_new_features, fn_kwargs = {'max_len': np.max(self.dataset['len_speech'])})

        if drop_columns is not None:
            print_if_debug('removing user-picked columns...', self.cc.DEBUG)
            assert isinstance(drop_columns, list) or isinstance(drop_columns, str)
            if isinstance(drop_columns, str): self.dataset = self.dataset.remove_columns([drop_columns])
            elif isinstance(drop_columns, list): self.dataset = self.dataset.remove_columns(drop_columns)
        self.dataset = self.dataset.remove_columns([self.feature, 'speech', 'len_speech', 'sampling_rate'])

        if target_processing is not None:
            print_if_debug('target processing... (is ON)', self.cc.DEBUG)
            assert isinstance(target_processing, dict)
            assert ['fn', 'kwargs'] == list(target_processing.keys()) 
            assert isinstance(target_processing['fn'], type(lambda x: None)) and\
                   isinstance(target_processing['kwargs'], dict)

            self.dataset = self.dataset.map(target_processing['fn'], fn_kwargs = target_processing['kwargs'])

        if save_path is not None:
            assert isinstance(save_path, str) 
            print_if_debug('saving files...', self.cc.DEBUG)
            if self.lang is None: self.lang = "en"
            self.dataset.save_to_disk(save_path + self.feature + "_" + self.lang + "_dataset")
        print('done')
        return self
    
    def run_probing(self, probing_taskk: Prober, probing_fn, layers: list, enable_grads = False, use_variational: bool = False, init_strategy: str = None, plotting_fn: Callable = None, 
                    save_checkpoints: bool = False, plotting_config: dict = None, **kwargs):
        """Main probing runner
        Args:
           probing_fn, init_strategy -- look at Prober docs
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
        probing_task = probing_taskk(self.model_path, self.writer, data = self.dataset, init_strategy = init_strategy)
        probing_task.get_resources(load_data = False, batch_size = self.cc.BATCH_SIZE, **kwargs)
       

        probing_results = probing_task.make_probe(probing_fn, use_variational = use_variational, enable_grads = enable_grads, layers = layers, 
                                                  save_outputs = save_checkpoints, task_title = plotting_config['title'])
        
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