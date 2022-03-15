stops =  ["b", "d", "g", "p", "t", "k", "dx", "q"]  
closed_stops = ["bcl", "dcl", "gcl", "pcl", "tck", "kcl"]   
####################################################################################
affricates = ["jh", "ch"]
closed_affricates = ["dcl", "tcl"]
####################################################################################
fricatives = ["s", "sh", "z", "zh", "f", "th"]
####################################################################################
nasals = ["m", "n", "ng", "em", "en", "eng", "nx"]

####################################################################################
semivowels_glides = ["l", "r", "w", "y", "hh", "hv", "el"]
####################################################################################
vowels = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
          "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]
####################################################################################
others = ["pau", "h#"]

from .base.processing import DatasetProcessor, Processor
from .base.utils import print_if_debug
from datasets import Dataset
from .tokenizers import Wav2Vec2OProcessor
from typing import Optional, Union, Dict, List
from collections import Callable
import os
import numpy as np
from copy import deepcopy

#TODO: fix docs

def comparison_dict(feature_lists: List[List], only_custom_features: bool = False):
    """Create a dict to compare two feature lists"""
    count = 0
    d = {d1_el: count for d1_el in feature_lists[0]}
    for dl in feature_lists[1:]:
        count += 1 
        d.update({d2_el: count for d2_el in dl})
    if only_custom_features: d.update({"other": count + 1})
    return d    

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
            self.task_data = self.task_data.filter(lambda example: len(example["speech"]) > 0)

        print_if_debug('encoding features...', self.cc.DEBUG)
        self._filter_data(self.tok2label, self.only_custom_features)
        self.task_data = self.task_data.map(encode_labels, fn_kwargs = {'feature_column': self.feature_column})
        print_if_debug('processing features...', self.cc.DEBUG)

        # return None
        self.task_data = self.task_data.map(self.tokenizer, fn_kwargs = {'data_column': 'speech', "feature_column": self.feature_column,
                                                                                'max_len': np.max(self.task_data['len_speech'])})

        if drop_columns is not None:
            print_if_debug('removing user-picked columns...', self.cc.DEBUG)
            assert isinstance(drop_columns, list) or isinstance(drop_columns, str)
            if isinstance(drop_columns, str): self.task_data = self.task_data.remove_columns([drop_columns])
            elif isinstance(drop_columns, list): self.task_data = self.task_data.remove_columns(drop_columns)
        self.task_data = self.task_data.remove_columns(['speech', 'len_speech', 'sampling_rate'] + ([self.feature_column] if self.feature_column != "label" else []))

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


# ARPABET was invented for English.
# The standard dictionary written in ARPABET is the CMU dictionary.
# TIMIT is written in a variant of ARPABET that includes a couple
# of non-standard allophones, and most significantly, includes
# separate symbols for the closure and release portions of each stop.
#https://github.com/jhasegaw/phonecodes/blob/master/src/phonecode_tables.py


_tone2ipa = {
    'arz' : { '0':'', '1':'ˈ', '2':'ˌ' },
    'eng' : { '0':'', '1':'ˈ', '2':'ˌ' },
    'yue' : { '0':'', '1':'˥', '2':'˧˥', '3':'˧', '4':'˨˩', '5':'˩˧', '6':'˨' },
    'lao' : { '0':'', '1':'˧', '2':'˥˧', '3':'˧˩', '4':'˥', '5':'˩˧', '6':'˩' },
    'cmn' : { '0':'', '1':'˥', '2':'˧˥', '3':'˨˩˦', '4':'˥˩', '5':'' },
    'spa' : { '0':'', '1':'ˈ', '2':'ˌ' },
    'vie' : { '0':'', '1':'˧', '2':'˨˩h', '3':'˧˥', '4':'˨˩˨', '5':'˧ʔ˥', '6':'˧˨ʔ' },
}
       
_arpabet2ipa = {
    'AA':'ɑ',
    'AE':'æ',
    'AH':'ʌ',
    'AH0':'ə',
    'AO':'ɔ',
    'AW':'aʊ',
    'AY':'aɪ',
    'EH':'ɛ',
    'ER':'ɝ',
    'ER0':'ɚ',
    'EY':'eɪ',
    'IH':'ɪ',
    'IH0':'ɨ',
    'IY':'i',
    'OW':'oʊ',
    'OY':'ɔɪ',
    'UH':'ʊ',
    'UW':'u',
    'B':'b',
    'CH':'tʃ',
    'D':'d',
    'DH':'ð',
    'EL':'l̩ ',
    'EM':'m̩',
    'EN':'n̩',
    'F':'f',
    'G':'ɡ',
    'HH':'h',
    'JH':'dʒ',
    'K':'k',
    'L':'l',
    'M':'m',
    'N':'n',
    'NG':'ŋ',
    'P':'p',
    'Q':'ʔ',
    'R':'ɹ',
    'S':'s',
    'SH':'ʃ',
    'T':'t',
    'TH':'θ',
    'V':'v',
    'W':'w',
    'WH':'ʍ',
    'Y':'j',
    'Z':'z',
    'ZH':'ʒ'
}
_arpabet2ipa.update(_tone2ipa['eng'])   # Add the English stress labels
_arpabet_vowels=set((k for k in _arpabet2ipa.keys() if k[0] in 'AEIOU'))

_ipa2arpabet = { v: k for k, v in _arpabet2ipa.items() }
_ipa2tone = {l:{v:k for k,v in d.items()} for l,d in _tone2ipa.items()}

_timit2ipa = _arpabet2ipa.copy()
_timit2ipa.update({
    'AX':'ə',
    'AX-H':'ə̥',
    'AXR':'ɚ',
    'B':'',
    'BCL':'b',
    'D':'',
    'DCL':'d',
    'DX':'ɾ',
    'ENG':'ŋ̍',
    'EPI':'',
    'G':'',
    'GCL':'g',
    'HV':'ɦ',
    'H#':'',
    'IX':'ɨ',
    'KCL':'k',
    'K':'',
    'NX':'ɾ̃',
    'P':'',
    'PAU':'',
    'PCL':'p',
    'T':'',
    'TCL':'t',
    'UX':'ʉ',
})

def convert_timit2ipa(batch):
    batch["ipa"] = _timit2ipa[batch["utterance"].upper()]
    return batch

def convert():
    from datasets import load_from_disk
    d = load_from_disk("../phonetic_set")
    d = d.map(convert_timit2ipa)
    d.save_to_disk("../phonetic_set")

