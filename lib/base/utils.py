
_lang = lambda l: "en" if l is None else l
print_if_debug = lambda x, flag: print(x) if flag else None
f_set = lambda col: {v: k for k, v in enumerate(list(set(col)))}

import re
import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
import json
import os
import sys
from .constants import Constants
from datasets import ReadInstruction, DatasetDict, Dataset, concatenate_datasets
from typing import Tuple, Union
from copy import deepcopy
from collections import OrderedDict

def copy_state_dict(model_from: OrderedDict, model_to: OrderedDict, keys2copy: list):
  """Copies keys from model_from.state_dict() to model_to.state_dict()
  keys2copy (list): list of keys copy between models
  """
  for key in keys2copy: model_to[key] = deepcopy(model_from[key])
  return model_to
  
def test_ipkernel():
    """A bool helper to know, whether the code is running in Jupyter notebook or not."""
    return 'ipykernel_launcher.py' in sys.argv[0] 

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower().strip()
    return batch


def pos_tagging_fn(batch):
    batch['pos_tag'] = nltk.pos_tag([batch['utterance']])[0][1]
    return batch

def random_labeling_fn(batch, label: str = 'pos_tag', n_classes: int = 10):
    batch[label + "_random"] = str(np.random.randint(0, n_classes))
    return batch

def label_batch(batch):
    batch = pos_tagging_fn(batch)
    batch = random_labeling_fn(batch)
    return batch

def _make_directory_structure():
    cc = Constants
    if not os.path.exists(cc.GRAPHS_PATH) or not os.path.exists(os.path.join(cc.GRAPHS_PATH, cc.TODAY)): os.makedirs(os.path.join(cc.GRAPHS_PATH, cc.TODAY))
    if not os.path.exists(cc.LOGGING_DIR): os.makedirs(cc.LOGGING_DIR)
    if not os.path.exists(cc.PROFILING_DIR) and cc.PROFILING: os.makedirs(cc.PROFILING_DIR)

def _check_download(downloaded: str, path: str) -> str:
    code = os.system(f"wget -O {path} {downloaded}")
    if code == 0: return "succeed"
    else: raise ValueError(f"os.system failed with code = {code}")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DatasetSplit(object):
    def __init__(self, split_str: Union[str, Tuple]) -> None:
        """A little helper for HuggingFace dataset splits"""
        self.expr = split_str

    def split_percent(self, data_split: str, fr0m: int, t0: int):
        self.expr = self.expr + (ReadInstruction(data_split, from_ = fr0m, to = t0, unit='%'), ) 
    
    def split_str(self, dataset: DatasetDict) -> Dataset:
        """A method that applies a given split expression to the dataset
        Args:
            dataset, DatasetDict: huggingface datasetdict
        Returns:
            Dataset, a prepocessed dataset
        """
        if isinstance(self.expr, str):
            if self.expr == "all": return concatenate_datasets([dataset[s] for s in dataset.keys()])
            else: return concatenate_datasets([dataset[s] for s in self.expr.split(";")])
        else: return NotImplementedError("")