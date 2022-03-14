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

    MODELS_PATH ={"common_voice": {"ru": "anton-l/wav2vec2-large-xlsr-53-russian",
                                "fr": "facebook/wav2vec2-large-xlsr-53-french",
                                "de": "facebook/wav2vec2-largee-xlsr-53-german",
                                "es": "facebook/wav2vec2-large-xlsr-53-spanish"},
                "timit_asr": {"None": "elgeish/wav2vec2-large-lv60-timit-asr"},
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



top_contituencies = [
    "ADVP_NP_VP_.",
    "CC_ADVP_NP_VP_.",
    "CC_NP_VP_.",
    "IN_NP_VP_.",
    "NP_ADVP_VP_.",
    "NP_NP_VP_.",
    "NP_PP_.",
    "NP_VP_.",
    "OTHER",
    "PP_NP_VP_.",
    "RB_NP_VP_.",
    "SBAR_NP_VP_.",
    "SBAR_VP_.",
    "S_CC_S_.",
    "S_NP_VP_.",
    "S_VP_.",
    "VBD_NP_VP_.",
    "VP_.",
    "WHADVP_SQ_.",
    "WHNP_SQ_.",
]