stops =  ["b", "d", "g", "p", "t", "k", "dx", "q"]  
closed_stops = ["bcl", "dcl", "gcl", "pcl", "tck", "kcl"]   
####################################################################################
vowels = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
          "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]
####################################################################################
others = ["pau", "h#"]

from .base.processing import DatasetProcessor, Processor
from .base.utils import print_if_debug
from datasets import Dataset
from .tokenizers import Wav2Vec2OProcessor
from typing import Optional, Union, Dict
from collections import Callable
import os
import numpy as np

#TODO: fix docs

class ASRDatasetProcessor(DatasetProcessor):
    """Dataset wrapper for Huggingface ASR datasets"""
    def __init__(self, dataset_type: str, model_path: Union[str, Dict], feature_column: str, tokenizer: Optional[Callable] = None,
                 dataset: Dataset = None, f_set: dict = None, only_custom_features: bool = False) -> None:
        super().__init__(dataset_type, model_path, filepath = os.curdir, dataset_name = dataset_type, 
                        feature_column = feature_column, tokenizer = tokenizer, f_set = f_set, 
                        only_custom_features = only_custom_features)
        self.task_data = dataset

    def get_data(self):
        """A method to define `self.dataset` which doesnt come from pipeline. Override it."""
 
    def _filter_data(self, own_feature_set: dict, only_custom_features: bool) -> None:
        """
        Args: 
          own_feature_set, dict (format {feat: int}): own mapping to probing labels
                                                    default = None
          only_custom_features, bool: flag whether to use only custom features or add "other" ground class,
                                      active only with own_feature_set, default = True

        """
        self.task_data = self.task_data.filter(lambda example: len(example[self.feature_column].strip()) > 0)
        if own_feature_set is None: self.f_set = {v: k for k, v in enumerate(list(set(self.task_data[self.feature_column])))}
        else: 
            assert isinstance(own_feature_set, dict)
            self.f_set = own_feature_set            
            if not only_custom_features:
                self.f_set["other"] = np.max(list(self.f_set.values())) + 1
                def foo(batch):
                    if batch[self.feature_column] not in self.f_set.keys(): batch[self.feature_column] = "other"
                    return batch                    
                self.task_data = self.task_data.map(foo)
            else: self.task_data = self.task_data.filter(lambda example: example[self.feature_column] in self.f_set)

    def process_dataset(self, preprocessing_fn: Callable, drop_columns: list = None, target_processing: Callable = None, 
                              _save_to_disk: bool = False) -> Dataset:
        """
        Args:
            preprocessinf_fn, callable object: prerpocessing dataset function to load all audio, should return the same self.dataset
                                               but with 'speech', 'len_speech', 'sampling_rate' columns
            drop_columns, list: optional list of string-like columns to drop from the dataset
                                default = None
            target_processing, callable object: prerpocessing dataset function to speech transcript. Shouldn't change the dataset structure
                                                default = None
            _save_to_disk, bool: an optional flag whether to save the preprocessed dataset on disk or nor.
        """
        print_if_debug("downloading necessary staff...", self.cc.DEBUG)
        def encode_labels(example, feature_column: str):
            """Label Encoder
            """
            example["label"] = self.f_set[example[feature_column]]
            
            return example

        print_if_debug('reading files...', self.cc.DEBUG)
        if preprocessing_fn is not None:
            self.task_data = self.task_data.map(preprocessing_fn, fn_kwargs = {'feature_column': self.feature_column}, disable_nullable = False)
        print_if_debug('encoding features...', self.cc.DEBUG)
        self._filter_data(self.tok2label, self.only_custom_features)
        self.task_data = self.task_data.map(encode_labels, fn_kwargs = {'feature_column': self.feature_column})
        print_if_debug('processing features...', self.cc.DEBUG)
        self.task_data = self.task_data.map(self.tokenizer, fn_kwargs = {'data_column': 'speech', "feature_column": self.feature_column,
                                                                                'max_len': np.max(self.task_data['len_speech'])})

        if drop_columns is not None:
            print_if_debug('removing user-picked columns...', self.cc.DEBUG)
            assert isinstance(drop_columns, list) or isinstance(drop_columns, str)
            if isinstance(drop_columns, str): self.task_data = self.task_data.remove_columns([drop_columns])
            elif isinstance(drop_columns, list): self.task_data = self.task_data.remove_columns(drop_columns)
        self.task_data = self.task_data.remove_columns([self.feature_column, 'speech', 'len_speech', 'sampling_rate'])

        if target_processing is not None:
            print_if_debug('target processing... (is ON)', self.cc.DEBUG)
            assert isinstance(target_processing, dict)
            assert ['fn', 'kwargs'] == list(target_processing.keys()) 
            assert isinstance(target_processing['fn'], Callable) and\
                   isinstance(target_processing['kwargs'], dict)

            self.task_data = self.task_data.map(target_processing['fn'], fn_kwargs = target_processing['kwargs'])
            self.task_data.set_format(type = 'torch', columns = ['input_values', 'attention_mask', 'label'])

            if _save_to_disk: self.task_data.save_to_disk(self.dname)
        return self.task_data