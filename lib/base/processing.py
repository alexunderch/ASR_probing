from .constants import Constants
from collections import Callable

from typing import Union,  Dict
from datasets import Dataset, DatasetDict
from typing import List, Optional


class Processor(object):
    """A base processor class. It is needed to wrap a tokenizer"""
    def __init__(self, model_path: str, tokenizer: Callable = None) -> None:
        """Args:
            model_path, str: a path to pretrained HuggingFace tokenizer checkpoint
            tokenizer, Callable: a tokenizer class of HuggingFace transformers library
                                 default = None
        """
        self.cc = Constants

        if model_path: self.tokenizer = tokenizer.from_pretrained(model_path, cache_dir = self.cc.CACHE_DIR)
    def __call__(self, batch, max_len: int, data_column: str = "data"): 
        """Preprocessing the given features with padding to maximum lenght"""
        raise NotImplementedError("")

class DatasetProcessor(object):
    """Base class for dataset processing. Outputs a HiggingFace-formatted dataset"""
    def __init__(self, dataset_type: str, 
                       model_path: Union[str, Dict],
                       filepath: str, dataset_name: str, 
                       feature_column: str, tokenizer: Optional[Callable] = None, 
                       f_set: dict = None, only_custom_features: bool = False) -> None:

        """Args:
            dataset_type, str: one of precomputed dataset_types: ['senteval', 'person', 'conn', 'DiscoEval', 'PDTB', 'huggingface', 'common_voice', 'timit_asr']
            model_path, str: path to a pretained tokenizer on HuggingFace Hub
            filepath, str: where to save a dataset
            dataset_name, str: a name for dataset
            feature_column, str: a column name where the labels can be found
            tokenizer, Processor: a tokenizer to process inputs
        """
        supported_datasets = [None, 'senteval', 'person', 'conn', 'DiscoEval', 'PDTB', 'huggingface', 'common_voice', 'timit_asr']
        assert dataset_type in supported_datasets, "no other types are not currently supported"
        self.dtype = dataset_type
        self.tokenizer = tokenizer(model_path = model_path)
        self.fpath = filepath
        self.mpath = model_path
        self.dname = dataset_name
        self.feature_column = feature_column
        self.maxlen = 0 
        self.tok2label = f_set
        self.only_custom_features = only_custom_features
        self.task_data = {'train': {'data': [], self.feature_column: []},
                        'dev': {'data': [], self.feature_column: []},
                        'test': {'data': [], self.feature_column: []}}
        self.cc = Constants
    
    def process_dataset(self, data_col: Union[str, List] = "data", load_from_disk = False) -> Union[Dataset, DatasetDict]:
        """A main processing function of the class.
        Args: 
            data_col, str: a column with necessary modality
            load_from_disk, bool: whether to load the data from disk or not
                                  default = False
        """

        raise NotImplementedError("")
