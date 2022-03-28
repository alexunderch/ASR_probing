from torchaudio import load, transforms
import re
import pickle
from transformers import Wav2Vec2Processor
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from  IPython.display import clear_output
from .phoneme_utils import _timit2ipa
from .base.constants import Constants
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

cc = Constants
def make_probing(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, shuffle = True, random_state=42)
    clf = SGDClassifier(max_iter = 1000, tol = 1e-2, random_state = 42)
    clf.fit(X_train, y_train)
    # acc = accuracy_score(clf.predict(X_test), y_test)
    f1 = f1_score(clf.predict(X_test), y_test, average = 'weighted')
    del clf
    return f1

##common_voice
def prepare_probing_task(batch, feature_column: str):
    sp, sr = torch.tensor(batch['audio']['array']), batch['audio']['sampling_rate']
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])
    return batch


def prepare_probing_task_(batch, feature_column: str):
    frame_offset, num_frames = 16000, 16000
    sp, sr = load(batch["path"], frame_offset = frame_offset, num_frames = num_frames)
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])
    return batch


##timit
def prepare_probing_task_timit_fast(batch, feature_column: str):
    frame_offset, num_frames = 16000, 16000
    sp, sr = load(batch["file"], frame_offset = frame_offset, num_frames = num_frames)
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])

    return batch

def prepare_probing_task_timit(batch, feature_column: str):
    sp, sr = load(batch["file"])
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])

    return batch

metadata = pd.read_csv(cc.TIMIT_METADATA_PATH)
def prepare_probing_task_timit_2(batch, feature_column: str):
    """ Adding new features from  dataset's metadata files
    """
    frame_offset, num_frames = 16000, 16000
    sp, sr = load(batch["file"], frame_offset = frame_offset, num_frames = num_frames)
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])
    val = metadata[metadata["id"] == batch["speaker_id"]][feature_column].values
    batch[feature_column] = str(val[0]) if len(val) else "other" 
    return batch

import re

def remove_special_characters(batch, text_column: str = "text"):
    chars_to_ignore_regex = '[\s\,\?\.\!\-\;\:\"]'
    tmp = re.sub(chars_to_ignore_regex, '', batch[text_column]).lower()
    all_symbols = []
    for word in tmp: all_symbols.extend(list(word))
    batch["len_text"] = len(all_symbols)
    batch[text_column] = " ".join(all_symbols)
    return batch

def ipa_processing_timit(batch, target_vocab: dict, text_column: str = 'text'):
    tmp = batch['phonetic_detail']['utterance']
    batch["len_text"] = len(tmp)
    totv = lambda x: str(target_vocab[x]) if x in list(target_vocab.keys()) else "<unk>"
    batch[text_column] = "".join([(totv(_timit2ipa[l.upper()]) if l.upper() in list(_timit2ipa.keys()) else l )for l in tmp])
    return batch
