from typing import Union,  Dict
from .base.constants import Constants
from .base.utils import print_if_debug, label_batch, _check_download
from .base.processing import Processor, DatasetProcessor


from torchaudio import load, transforms
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from collections import Callable
import json
import os
import numpy as np
from copy import deepcopy
from typing import List, Optional

class EmbeddingTensorDataset(DatasetProcessor):
    def __init__(self,  feature_column: str, filepath: str = None, dataset_name: str = None, tokenizer: Optional[Callable] = None, 
                       dataset: Dataset = None, f_set: dict = None, only_custom_features: bool = False) -> None:
        super().__init__(dataset_type = "embedding_list", model_path = None, 
                         filepath = filepath, dataset_name = dataset_name, 
                         feature_column = feature_column, tokenizer = tokenizer, 
                         f_set = f_set, only_custom_features = only_custom_features)
        if dataset: self.task_data = dataset
 
    def _filter_data(self, own_feature_set: dict, only_custom_features: bool) -> None:
        """
        Args: 
          own_feature_set, dict (format {feat: int}): own mapping to probing labels
                                                    default = None
          only_custom_features, bool: flag whether to use only custom features or add "other" ground class,
                                      active only with own_feature_set, default = True

        """
        if isinstance(self.task_data[self.feature_column][0], str):
            self.task_data = self.task_data.filter(lambda example: len(example[self.feature_column].strip()) > 0)
        self.f_set = None
        if own_feature_set is None: self.f_set = {v: k for k, v in enumerate(list(set(self.task_data[self.feature_column])))}
        else: 
            assert isinstance(own_feature_set, dict)
            self.f_set = deepcopy(own_feature_set)            
            if not only_custom_features:
                self.f_set["other"] = np.max(list(self.f_set.values())) + 1
                def foo(batch):
                    if batch[self.feature_column] not in self.f_set.keys(): batch[self.feature_column] = "other"
                    return batch                    
                self.task_data = self.task_data.map(foo)
            else: self.task_data = self.task_data.filter(lambda example: example[self.feature_column] in self.f_set)

    def process_dataset(self, preprocessing_fn: Callable, drop_columns: list = None, _save_to_disk: bool = False, data_column: str = "data") -> Dataset:
        """
        Args:
            preprocessinf_fn, callable object: prerpocessing dataset function to load all audio, should return the same self.dataset
                                               but with 'speech', 'len_speech', 'sampling_rate' columns
            drop_columns, list: optional list of string-like columns to drop from the dataset
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
            print_if_debug('processing features...', self.cc.DEBUG)
            self.task_data = self.task_data.map(preprocessing_fn, fn_kwargs = {'feature_column': self.feature_column}, disable_nullable = False)

        print_if_debug('encoding features...', self.cc.DEBUG)
        self._filter_data(self.tok2label, self.only_custom_features)
        self.task_data = self.task_data.map(encode_labels, fn_kwargs = {'feature_column': self.feature_column})

        self.task_data = self.task_data.map(self.tokenizer, fn_kwargs = {"data_column": data_column})

        if drop_columns is not None:
            print_if_debug('removing user-picked columns...', self.cc.DEBUG)
            assert isinstance(drop_columns, list) or isinstance(drop_columns, str)
            if isinstance(drop_columns, str): self.task_data = self.task_data.remove_columns([drop_columns])
            elif isinstance(drop_columns, list): self.task_data = self.task_data.remove_columns(drop_columns)

        
        self.task_data.set_format(type = 'torch', columns = ['input_values', 'label'])

        if _save_to_disk: self.task_data.save_to_disk(self.dname)
        return self.task_data
