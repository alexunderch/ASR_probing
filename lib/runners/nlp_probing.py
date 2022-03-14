from ..linguistic_utils import NLPDatasetProcessor
from ..tokenizers import T5Processor, BertProcessor
from ..probers import BertOProber, T5EncoderProber, T5EncoderDecoderProber
from ..base.processing import Processor
from ..pipeline import Probing_pipeline
from IPython.display import clear_output
from ..clf import ProberModel
from ..base.constants import Constants
from ..base.prober import Prober
from ..base.utils import _lang, _make_directory_structure, DatasetSplit
from collections import Callable
from typing import Union, Dict, List
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class SimpleNLPPipeline(object):
    def __init__(self, model2probe: Prober, 
                        dataset_name: str,
                        morph_call: str,
                        feature: str,
                        layers: Union[List, Dict],
                        tokenizer: Processor,
                        data_column: str = 'text',
                        model_path: str = None,
                        download_data: bool = True,
                        dataset_split: str = "test",
                        data_path: str = None,
                        checkpoint: Union[str, Dict] = None,
                        model_init_strategies: list = [None] + ["full"], 
                        use_mdl: bool = False,
                        device: torch.device = torch.device('cpu'), 
                        probing_fn: torch.nn.Module = ProberModel,
                        enable_grads = False,
                        save_checkpoints: bool = False, 
                        return_results: bool = False,
                        is_prepared: bool = False,
                        **kwargs):
        cc = Constants
        if model_path is None: model_path = cc.MODELS_PATH[dataset_name]["None"][0]

        data_proc = NLPDatasetProcessor(dataset_type = morph_call, 
                                        model_path = model_path, 
                                        filepath = data_path, 
                                        dataset_name = dataset_name, 
                                        feature_column = feature, tokenizer = tokenizer)
        
        data_proc.download_data(download = download_data, is_prepared = is_prepared)
        data_proc.process_dataset(data_col = data_column, _save_to_disk = True) 

        self.results = []
        layers = list(sorted(layers)) if isinstance(layers, list) else layers
        _ = _make_directory_structure()

        for init_strategy in model_init_strategies:
            title = dataset_name + "_" 'lang=en_' + feature + "_task_random=" + str(init_strategy) +\
                    "_grads="  +str(enable_grads) + "_variational=" + str(use_mdl)
            writer = SummaryWriter(os.path.join(cc.LOGGING_DIR, title, 
                                  "layers={}-{}".format(layers[0], layers[-1]) if isinstance(layers, list) else f"layers={layers}"))
            pipe = Probing_pipeline(writer = writer, device = device,
                                    feature = feature, model_path = model_path, 
                                    lang = None, split = DatasetSplit(dataset_split))
            pipe.disable_cache()
            if not download_data: assert isinstance(data_path, str)
            pipe.load_data(from_disk = True, data_path = dataset_name, **kwargs)

            print("The task title:", title)
            print(f"The features are: {data_proc.tok2label if data_proc.tok2label else pipe.f_set}")
            res = pipe.run_probing(model2probe, probing_fn, layers = layers, enable_grads = enable_grads, 
                                use_variational = use_mdl, 
                                checkpoint = checkpoint,
                                save_checkpoints = save_checkpoints, 
                                init_strategy = init_strategy, 
                                plotting_config = {"title": title,
                                                    "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))})
            pipe.cleanup()
            if return_results: self.results.append(res)
        if not cc.DEBUG: clear_output(wait = True) 



