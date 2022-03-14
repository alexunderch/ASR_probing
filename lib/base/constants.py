from datetime import datetime
import os
class Constants:
    TODAY = str(datetime.now().strftime('%Y-%m-%d')) 
    POOLING_TO = 4
    BATCH_SIZE = 45
    N_EPOCHS = 10
    MAX_LEN = 13000 #for linguistic dataset
    DEBUG = False 
    PROFILING = False 

    MODELS_PATH ={"common_voice": {"ru": ["anton-l/wav2vec2-large-xlsr-53-russian", "1.10.0"],
                                "fr": ["facebook/wav2vec2-large-xlsr-53-french", "1.10.0"],
                                "de": ["facebook/wav2vec2-largee-xlsr-53-german", "1.10.0"],
                                "es": ["facebook/wav2vec2-large-xlsr-53-spanish", "1.10.0"]},
                "timit_asr": {"None": ["elgeish/wav2vec2-large-lv60-timit-asr", "1.10.0"]},
                "bert": {"None": "bert-large-cased"},
                "t5": {"None": "t5-small"},
                "roberta": {"None": "roberta-large-cased"}
                }

    TIMIT_METADATA_PATH = os.path.join(os.path.curdir, 'timit_features_proc.csv')
    LOGGING_DIR = os.path.join(os.path.curdir, 'tensorboards')
    GRAPHS_PATH = os.path.join(os.path.curdir, 'jsons')
    CHECKPOINTING_DIR = os.path.curdir
    PROFILING_DIR = os.path.join(os.path.curdir, 'tensorboards', 'profiling')
    CACHE_DIR = os.path.join(os.path.curdir, 'cache')




