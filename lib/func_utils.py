from torchaudio import load, transforms
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  IPython.display import clear_output

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
    frame_offset, num_frames = 16000, 16000
    sp, sr = load(batch["path"], frame_offset = frame_offset, num_frames = num_frames)
    resampler = transforms.Resample(sr, 16000)
    batch['speech'] = resampler(sp).squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch['len_speech'] = len(batch['speech'])
    return batch


##timit
def prepare_probing_task_timit(batch, feature_column: str):
    frame_offset, num_frames = 16000, 16000
    sp, sr = load(batch["file"], frame_offset = frame_offset, num_frames = num_frames)
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


def plotting_fn(data, config: dict):
    if not cc.DEBUG: clear_output(wait = True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(config['title'])
    used_metrics = config['metrics']
    if len(used_metrics) == 1: data = np.array(data)[..., np.newaxis]
    assert len(used_metrics) == data.shape[1]
    for ind in range(data.shape[1]): 
        ax.plot(np.arange(0, len(data), 1), data[:, ind], marker = '.', color = 'red', lw = 2, label = used_metrics[ind])
    ax.legend(loc = 'best')
    ax.set_ylabel('metrics'); ax.set_xlabel('#layer')
    ax.grid(True)
    if config['save_path'] is not None:
        assert isinstance(config['save_path'], str)
        pickle.dump(fig, open(config['save_path'] + config['title'] + '.pickle', 'wb'))

