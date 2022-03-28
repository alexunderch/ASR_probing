from concurrent.futures import process
from .base.constants import Constants
from transformers import BertTokenizer, Wav2Vec2Processor, T5Tokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from .base.processing import Processor
import json
from .base.utils import print_if_debug


class BertProcessor(Processor):
    def __init__(self, model_path) -> None:
        super().__init__(tokenizer = BertTokenizer, model_path = model_path)
    def __call__(self, batch, max_len: int, data_column: str = "data"):
        """Preprocessing text features with padding to maximum lenght"""
        inputs = self.tokenizer(batch[data_column], return_tensors = 'pt',  
                                    max_length = max_len, truncation = True, padding = 'max_length')
        batch['input_values'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        return batch

class Wav2Vec2OProcessor(Processor):
    def __init__(self, model_path) -> None:
        super().__init__(tokenizer = Wav2Vec2Processor, model_path =  model_path)
    def __call__(self, batch, max_len: int, feature_column: str, data_column: str = "speech"):
        """Preprocessing audio features with padding to maximum lenght"""
        inputs = self.tokenizer(batch[data_column], sampling_rate = batch["sampling_rate"], return_tensors = "pt", 
                                padding = 'max_length', truncation = 'max_length', max_length = max_len)
        batch['input_values'] = inputs.input_values
        batch['attention_mask'] = inputs.attention_mask

        return batch

class Wav2Vec2PProcessor(Processor):
    def __init__(self, model_path: str) -> None:
        # super().__init__(tokenizer = Wav2Vec2Processor, model_path =  None)
        self.cc = Constants
        self.vocab = dict()
        self.encoding_vocab = None
        vocab_dict = {'</s>': 2, '<pad>': 0, '<s>': 1, '<unk>': 3, '|': 4}

        with open('vocab.json', 'w') as vocab_file: json.dump(vocab_dict, vocab_file)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size = 1, sampling_rate = 16000, padding_value = 0.0, 
                                                     do_normalize = True, return_attention_mask = True)
        self.tokenizer =  Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = Wav2Vec2CTCTokenizer('vocab.json'))
    def update_vocab(self, v: dict): 
        """Updating vocab with new tokens for CTC probing encoding"""
        self.encoding_vocab = v
        self.vocab.update(v)
        # self.tokenizer.tokenizer.add_tokens(list(set(self.vocab.keys())))
        self.tokenizer.tokenizer.add_tokens(list(set(self.vocab.values())))

        print_if_debug(str(self.tokenizer.tokenizer.get_vocab()), self.cc.DEBUG)
        
    def __call__(self, batch, max_len: int, feature_column: str, data_column: str = "speech", labels_max_len: int = 100):
        """Preprocessing audio features with padding to maximum lenght"""
        inputs = self.tokenizer(batch[data_column], sampling_rate = batch["sampling_rate"], return_tensors = "pt", 
                                padding = 'max_length', truncation = 'max_length', max_length = max_len)
        batch['input_values'] = inputs.input_values
        batch['attention_mask'] = inputs.attention_mask
        
        with self.tokenizer.as_target_processor():
            labels_batch = self.tokenizer(batch[feature_column],
                                            padding = "max_length",
                                            truncation = True,
                                            max_length =labels_max_len,
                                            return_tensors = "pt")
        batch['label'] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch


class T5Processor(Processor):
    def __init__(self, model_path: str) -> None:
        super().__init__(tokenizer = T5Tokenizer, model_path = model_path)
    def __call__(self, batch, max_len: int, data_column: str = "data"):
        """Preprocessing text features with padding to maximum lenght"""
        inputs = self.tokenizer(batch[data_column], return_tensors = 'pt',  
                                    max_length = max_len, truncation = True, padding = 'max_length')
        batch['input_values'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        return batch