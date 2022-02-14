from typing import Union,  Dict
from base.constants import Constants
from base.utils import print_if_debug, label_batch
from base.processing import Processor, DatasetProcessor


from torchaudio import load, transforms
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from collections import Callable
import json
import os
import pandas as pd
import numpy as np
from typing import List, Optional

modes = ['word_detail', 'phonetic_detail']
class LinguisticDataset:
    def __init__(self, mode: str, save_path: str = None):
        """Linguistic dataset wrapper for TIMIT_ASR dataset
        Args:
             mode, str: 'word_detail' or 'phonetic_detail'
             save_path, str: optional way to save probing data, active only if save_outputs = True;
                            default = None
        """
        assert mode in ['word_detail', 'phonetic_detail']
        self.mode = mode
        self.save_path = save_path
        self.D = load_dataset('timit_asr', split = 'test')
        self.data = []
    def format_dataset(self):
        """Making JSON for TIMIT_ASR dataset
        """
        def process_fn(batch):
            data = {'long': []}

            for i in range(len(batch[self.mode]['start'])): data['long'].append((batch[self.mode]['start'][i], 
                                                                                  batch[self.mode]['stop'][i]))
            data['path'] = batch['file']
            data['utterance'] = batch[self.mode]['utterance']
            return data
        print_if_debug('making json...', Constants.DEBUG)
        for batch in self.D: self.data.append(process_fn(batch)) 
        if self.save_path is not None: assert isinstance(self.save_path, str)
        else: self.save_path = '.'   
        with open(self.save_path + 'timit_'+ self.mode +'.json', 'w') as fout: json.dump(self.data, fout)

    def __call__(self, additional_preprocessing: Callable = None, debug: bool = False, take_n: int = None) -> Dataset:
        """
        Args:
            additional_preprocessing, callable object: function to make new labels from phonemes, eg. POS-tags
                                                      default  = None, 
            debug, bool: flag, default = False
            take_n, int: how many samples from original set to take
                        default   = 10000 (None)
        Returns: new Hugging Face dataset
        """
        self.format_dataset()
        dataset = Dataset.from_pandas(pd.read_json(self.save_path + 'timit_'+ self.mode +'.json'))
        def mapping_fn(batch):
            """Phonemes mapping
            """
            speech_array, sampling_rate = load(batch["path"])
            resampler = transforms.Resample(sampling_rate, 16000)
            batch["speech"] = speech_array.squeeze().numpy()
            batch['atoms'] = []
            for l in batch['long']: 
                batch['atoms'].append(batch['speech'][l[0]:l[1]])
            return batch

        dataset = dataset.map(mapping_fn)
        new_data = []
        for example in dataset:
            for ind in range(len(example['atoms'])):
              if  len(example['atoms'][ind]) > 0 and \
                  len(example['atoms'][ind]) <= Constants.MAX_LEN if self.mode == 'word_detail' else int(0.01 * Constants.MAX_LEN): 

                  new_data.append([example['atoms'][ind], 16000,
                                   len(example['atoms'][ind]),
                                   example['utterance'][ind]])

              else: print_if_debug(example['utterance'][ind] + " " + "{}".format(len(example['atoms'][ind])), debug)
        new_dataset = pd.DataFrame(new_data, columns = ['speech', 'sampling_rate', 'len_speech', 'utterance'])

        if take_n is not None: 
            assert isinstance(take_n, int)
            print_if_debug("taking a slice of {} elements".format(take_n), debug)
            new_dataset = new_dataset.sample(n = take_n, random_state = 42)

        new_dataset =  Dataset.from_pandas(new_dataset)
        if additional_preprocessing is not None:
            assert isinstance(additional_preprocessing, type(lambda x: None))
            print_if_debug("adding new features...", debug)
            new_dataset = new_dataset.map(additional_preprocessing, batched = False)
        return new_dataset

class NLPDatasetProcessor(DatasetProcessor): 
    def __init__(self, dataset_type: str, model_path: Union[str, Dict], filepath: str, dataset_name: str, feature_column: str, tokenizer: Optional[Processor] = None):
        super().__init__(dataset_type, model_path, filepath, dataset_name, feature_column, tokenizer)
    def download_data(self) -> str:
        if self.dtype == 'senteval': 
            assert self.fpath.endswith("tsv")
            os.system(f"wget -O {self.fpath} https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/past_present.txt")
        elif self.dtype == 'person': 
            assert self.fpath.endswith("csv") or self.fpath.endswith("tsv")
            os.system(f"wget -O {self.fpath} https://media.githubusercontent.com/media/morphology-probing/morph-call/main/data/morphosyntactic_values/english/person.tsv")
        elif self.dtype == 'conn': pass     
        elif self.dtype == 'DiscoEval': 
            assert isinstance(self.fpath, str) and self.fpath in ['DC', 'SP']
            os.makedirs(self.fpath, exist_ok=True)
            os.system(f"wget -O {os.path.join(self.fpath, 'train.txt')} https://raw.githubusercontent.com/ZeweiChu/DiscoEval/master/data/{self.fpath}/wiki/train.txt")
            os.system(f"wget -O {os.path.join(self.fpath, 'dev.txt')}  https://raw.githubusercontent.com/ZeweiChu/DiscoEval/master/data/{self.fpath}/wiki/valid.txt")
            os.system(f"wget -O {os.path.join(self.fpath, 'test.txt')}  https://raw.githubusercontent.com/ZeweiChu/DiscoEval/master/data/{self.fpath}/wiki/test.txt")
            self.load_DC_dataset(os.path.join(self.fpath, 'train.txt'), 'train')
            self.load_DC_dataset(os.path.join(self.fpath, 'dev.txt'), 'dev')
            self.load_DC_dataset(os.path.join(self.fpath, 'test.txt'), 'test')
        elif self.dtype == 'PDTB': pass
        else: print("nothing to download")

    def load_TXT_file(self):
        import io
        self.task_data = {'train': {'data': [], self.feature_column: []},
                              'dev': {'data': [], self.feature_column: []},
                              'test': {'data': [], self.feature_column: []}}
        tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        with io.open(self.fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                text = line[-1].split()
                self.maxlen = max(len(text), self.maxlen)

                self.task_data[tok2split[line[0]]]['data'].append(" ".join(text))
                self.task_data[tok2split[line[0]]][self.feature_column].append(line[1])

        labels = sorted(np.unique(self.task_data['train'][self.feature_column]))
        self.tok2label = dict(zip(labels, range(len(labels))))

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split][self.feature_column]):
                self.task_data[split][self.feature_column][i] = self.tok2label[y]

    def load_CSV_file(self, sep = ',', data_col: Union[str, List] = 'data', header: int = None):
        self.task_data = {'train': {'data': [], self.feature_column: []},
                                'dev': {'data': [], self.feature_column: []},
                                'test': {'data': [], self.feature_column: []}}
        tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        tmp_data = pd.read_csv(self.fpath, sep = sep, header = header)

        for abbr, split in tok2split.items(): 
            self.task_data[split] = tmp_data[tmp_data["subset"] == abbr][[data_col, self.feature_column]]
        
        self.maxlen = self.task_data["train"][data_col].map(lambda x: x.split(" ")).map(len).max() 

        labels = sorted(np.unique(self.task_data['train'][self.feature_column]))
        self.tok2label = dict(zip(labels, range(len(labels))))

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split][self.feature_column].values):
                self.task_data[split][self.feature_column].values[i] = self.tok2label[y]

    
    def load_DC_dataset(self, path: str, _split: str):
        import io
        with io.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                text = line[-1].split()
                if _split == "train": self.maxlen = max(len(text), self.maxlen)
                self.task_data[_split]['data'].append(" ".join(text))
                self.task_data[_split][self.feature_column].append(line[0])
        
        if _split == "train": 
            labels = sorted(np.unique(self.task_data[_split][self.feature_column]))
            self.tok2label = dict(zip(labels, range(len(labels))))
        for i, y in enumerate(self.task_data[_split][self.feature_column]):
            self.task_data[_split][self.feature_column][i] = self.tok2label[y]
    
    def loadHuggingFaceDataset(self, from_disk: bool = True):
        if from_disk: self.task_data = load_from_disk(dataset_path = self.fpath)
        else: self.task_data = load_dataset(path = self.dname)


    def process_dataset(self, data_col: Union[str, List] = "data", load_from_disk = False): 
        if not os.path.exists(self.fpath): _ = self.download_data() 
        if load_from_disk:
            self.task_data = self.loadHuggingFaceDataset(from_disk = load_from_disk)
            dataset = self.task_data.map(self.tokenizer, fn_kwargs = {'max_len': self.maxlen, 'data_column': data_col})
        else:
            dataset = DatasetDict()
            for k, v in self.task_data.items(): 
                dataset[k] = Dataset.from_dict(v)
                dataset[k] = dataset[k].map(self.tokenizer, fn_kwargs = {'max_len': self.maxlen, 'data_column': data_col} )
        dataset.set_format(type = 'torch', columns = ['input_values', 'attention_mask', 'label'])
        dataset.save_to_disk(self.dname)


def make_dataset() -> DatasetDict:
    lingusitic_dataset = DatasetDict()
    for mode in modes:
        lingusitic_dataset[mode] = LinguisticDataset(mode, 
                                                    save_path = ".")(
                                                    label_batch if mode == 'word_detail' else None,
                                                    debug = Constants.DEBUG,
                                                    take_n = None if mode == 'word_detail' else 15000 )
    return lingusitic_dataset