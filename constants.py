BATCH_SIZE = 50
N_EPOCHS = 10
MAX_LEN = 13000 #for linguistic dataset
DEBUG = False 
PROFILING = False 

MODELS_PATH ={"common_voice": {"ru": "anton-l/wav2vec2-large-xlsr-53-russian",
                               "fr": "facebook/wav2vec2-large-xlsr-53-french",
                               "de": "facebook/wav2vec2-large-xlsr-53-german",
                               "es": "facebook/wav2vec2-large-xlsr-53-spanish"},
              "timit_asr": {"None": "elgeish/wav2vec2-large-lv60-timit-asr"}}
TIMIT_METADATA_PATH = f"./timit_features_proc.csv"
LOGGING_DIR = "./tensorboards/"
GRAPHS_PATH = "./jsons/"
CHECKPOINTING_DIR = "./"
PROFILING_DIR = "./tensorboards/profiling/"
CACHE_DIR = './cache'
POOLING_TO = 4