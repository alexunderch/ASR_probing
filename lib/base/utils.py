
_lang = lambda l: "en" if l is None else l
print_if_debug = lambda x, flag: print(x) if flag else None
f_set = lambda col: {v: k for k, v in enumerate(list(set(col)))}

import re
import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
import json
import os
from .constants import Constants

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

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _make_directory_structure():
    cc =Constants
    if not os.path.exists(cc.GRAPHS_PATH) or not os.path.exists(os.path.join(cc.GRAPHS_PATH, cc.TODAY)): os.makedirs(os.path.join(cc.GRAPHS_PATH, cc.TODAY))
    if not os.path.exists(cc.LOGGING_DIR): os.makedirs(cc.LOGGING_DIR)
    if not os.path.exists(cc.PROFILING_DIR) and cc.PROFILING: os.makedirs(cc.PROFILING_DIR)
