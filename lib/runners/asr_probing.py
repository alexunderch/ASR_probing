from cmath import pi
from collections import Callable
from typing import Union, List, Dict
import os
import torch

from torch.utils import tensorboard
from ..base.constants import Constants
from ..func_utils import prepare_probing_task, prepare_probing_task_timit, prepare_probing_task_timit_2, prepare_probing_task_, ipa_processing_timit
from ..base.utils import _lang, _make_directory_structure, DatasetSplit, f_set
from ..base.processing import Processor
from ..phoneme_utils import ASRDatasetProcessor, comparison_dict
from ..phoneme_utils import *
from ..base.task import TaskTester
from ..clf import ProberModel
from ..probers import Prober, Wav2Vec2Prober
from ..pipeline import Probing_pipeline
from IPython.display import clear_output
from ..tokenizers import Wav2Vec2OProcessor, Wav2Vec2PProcessor


class SimpleASRPipeline(TaskTester):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self,  model2probe: Prober, 
                        dataset_language: List,
                        morph_call: str,
                        features: Union[List, str],
                        layers: Union[List, Dict],
                        tokenizer: Processor,
                        dataset_name: str = None,
                        model_path: str = None,
                        preprocessing_fn: Callable = prepare_probing_task,
                        dataset_split: str = "test",
                        data_path: str = None,
                        checkpoint: Union[str, Dict] = None,
                        model_init_strategies: list = [None] + ["full"], 
                        use_mdl: bool = False,
                        device: torch.device = torch.device('cpu'), 
                        probing_fn: torch.nn.Module = ProberModel,
                        enable_grads = False,
                        own_feature_set = None,
                        save_checkpoints: bool = False, 
                        poisoning_ratio: float = 0,
                        poisoning_mapping = None,
                        from_disk: bool = False,
                        use_ctc_objectve: bool = True,
                        **kwargs) -> dict:
        self.results = []
        layers = list(sorted(layers))   
        cc = Constants
        

        _ = _make_directory_structure()
        for lang in dataset_language:
            if model_path is None: model_path = cc.MODELS_PATH[morph_call][str(lang)][0] 
            for feature in features:
                
                for init_strategy in model_init_strategies:
                    title = (morph_call if dataset_name is None else dataset_name) +\
                             "_" + _lang(lang) + "_" + feature + "_task_random=" + str(init_strategy) +\
                            "_grads="  +str(enable_grads) + "_variational=" + str(use_mdl) + (("_poisoned=" + str(poisoning_ratio)) if poisoning_ratio > 0 else str())
                    writer = tensorboard.SummaryWriter(os.path.join(cc.LOGGING_DIR, title, "layers={}-{}".format(layers[0], layers[-1])))
                    pipe = Probing_pipeline(writer = writer, device = device,
                                            feature = feature, model_path = model_path, 
                                            lang = lang,  split = DatasetSplit(dataset_split))
                    pipe.disable_cache()
                    if from_disk: assert isinstance(data_path, str)
                    pipe.load_data(from_disk = from_disk, data_path = (morph_call if dataset_name is None else dataset_name) if not from_disk else data_path, revision = cc.MODELS_PATH[morph_call][str(lang)][-1])
                    data_proc = ASRDatasetProcessor(dataset_type=morph_call, model_path=model_path, 
                                                    feature_column=feature, tokenizer=tokenizer, dataset=pipe.dataset,
                                                    f_set = own_feature_set, only_custom_features = False, for_ctc = use_ctc_objectve)
                    
                   
                    pipe.dataset = data_proc.process_dataset(preprocessing_fn=preprocessing_fn,
                                                            target_processing = {'fn': ipa_processing_timit, 'kwargs':  {"target_vocab": own_feature_set}} if use_ctc_objectve else None,
                                                            _save_to_disk = False)
                    print("The task title:", title)
                    res = pipe.run_probing(model2probe, probing_fn, layers = layers, enable_grads = enable_grads, 
                                use_variational = use_mdl, 
                                checkpoint = checkpoint,
                                save_checkpoints = save_checkpoints, 
                                init_strategy = init_strategy, use_ctc_objective = use_ctc_objectve,
                                model_vocab = len(data_proc.tokenizer.tokenizer.tokenizer.get_vocab()),
                                plotting_config = {"title": title,
                                                    "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))},
                                poisoning_ratio = poisoning_ratio,
                                poisoning_mapping = poisoning_mapping)
                    pipe.cleanup()
                    self.results.append(res)
        if not cc.DEBUG: clear_output(wait = True) 
        return self.results


# class SimpleASRPipeline:
#     def __init__(self, model2probe: Prober,
#                        dataset_name: str,
#                        features: list,
#                        layers: list,
#                        dataset_language: list = [None],
#                        dataset_split: str = "test",
#                        from_disk: bool = False,
#                        prefix_data_path: str = None,
#                        model_init_strategies: list = [None] + ["full"], 
#                        use_variational: bool = False,
#                        device: torch.device = torch.device('cpu'), 
#                        preprocessing_fn: Callable = prepare_probing_task,
#                        probing_fn: torch.nn.Module = ProberModel,
#                        enable_grads = False,
#                        plotting_fn: Callable = None,
#                        save_checkpoints: bool = False, 
#                        save_preprocessed_data: str = None,
#                        own_feature_set: dict = None,
#                        poisoning_ratio: float = 0,
#                        poisoning_mapping = None,
#                        return_results: bool = False,
#                        **kwargs):
#         """A wrapper for the whole pipeline
#         Agrs:
#             model2probe: Prober class, an instance wrapper of model being probbed
#             dataset_name, str: name of used dataset, if `from_disk`== False, supported only "common_voice" and "timit_asr"
#             layers, list: list of layers to probe on (l >= 1 and l <= 24 for l in layers)
#             dataset_language, list: a list of supported dataset's languages
#                                     deafult (for timit_asr) = [None]
#             dataset_split, str: split of the used dataset
#                                 default = "test"
#             from_disk, bool: an optional flag where to take a data for probing (if False, data will be dowlnoaded 
#                              from HuggingFace hub)
#                              default = False
#             prefix_data_path, str: an optinal helpful string to find a dataset on the disk
#             model_init_strategies, list: list of probing initializers; supported regims are: 
#                                                                                             [None (only pretrained model),
#                                                                                              "full" (whole random init.),
#                                                                                              "encoder" (only random encoder)]
#                                          default = [None, "full"]
            
#             use_variational, bool: an optional flag, whether use MDL (True) or on ordinary logistic regression 
#                                     default = False
#             preprocessing_fn, a callable object: function for extracting audio from dataset
#                                                  default = None
#             probing_fn, torch.nn.Module class: probing model
#                                                defalut = ProberModel
            
#             enable_grads, bool: an optional flag, if backprop through any used model or not
#                                 default = False
#             own_feature_set: dict, an optional dict of hand-designed labels for probing,
#                                    default = None
            
#             return_results: bool: a flag
#             save_checkpoints, bool: an optional flag, whether to save checkpoints
#                                    defalult = False
            
#             other arguments are temporarily deprecated.
                                   
#         """
