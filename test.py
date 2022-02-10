from collections import Callable
import os
import torch
from torch.utils import tensorboard

from lib.constants import Constants
from lib.func_utils import prepare_probing_task, prepare_probing_task_timit, prepare_probing_task_timit_2, _lang
from lib.clf import ProberModel
from lib.probers import Prober, BertOProber, Wav2Vec2Prober
from lib.constants import Constants
from lib.pipeline import Probing_pipeline
from IPython.display import clear_output
class TaskTester:
    def __init__(self, model2probe: Prober,
                       dataset_name: str,
                       features: list,
                       layers: list,
                       dataset_language: list = [None],
                       dataset_split: str = "test",
                       from_disk: bool = False,
                       prefix_data_path: str = None,
                       model_init_strategies: list = [None] + ["full"], 
                       use_variational: bool = False,
                       device: torch.device = torch.device('cpu'), 
                       preprocessing_fn: Callable = prepare_probing_task,
                       probing_fn: torch.nn.Module = ProberModel,
                       enable_grads = False,
                       plotting_fn: Callable = None,
                       save_checkpoints: bool = False, 
                       save_preprocessed_data: str = None,
                       own_feature_set: dict = None,
                       poisoning_ratio: float = 0,
                       poisoning_mapping = None,
                       return_results: bool = False,
                       **kwargs):
        """A wrapper for the whole pipeline
        Agrs:
            model2probe: Prober class, an instance wrapper of model being probbed
            dataset_name, str: name of used dataset, if `from_disk`== False, supported only "common_voice" and "timit_asr"
            layers, list: list of layers to probe on (l >= 1 and l <= 24 for l in layers)
            dataset_language, list: a list of supported dataset's languages
                                    deafult (for timit_asr) = [None]
            dataset_split, str: split of the used dataset
                                default = "test"
            from_disk, bool: an optional flag where to take a data for probing (if False, data will be dowlnoaded 
                             from HuggingFace hub)
                             default = False
            prefix_data_path, str: an optinal helpful string to find a dataset on the disk
            model_init_strategies, list: list of probing initializers; supported regims are: 
                                                                                            [None (only pretrained model),
                                                                                             "full" (whole random init.),
                                                                                             "encoder" (only random encoder)]
                                         default = [None, "full"]
            
            use_variational, bool: an optional flag, whether use MDL (True) or on ordinary logistic regression 
                                    default = False
            preprocessing_fn, a callable object: function for extracting audio from dataset
                                                 default = None
            probing_fn, torch.nn.Module class: probing model
                                               defalut = ProberModel
            
            enable_grads, bool: an optional flag, if backprop through any used model or not
                                default = False
            own_feature_set: dict, an optional dict of hand-designed labels for probing,
                                   default = None
            
            return_results: bool: a flag
            save_checkpoints, bool: an optional flag, whether to save checkpoints
                                   defalult = False
            
            other arguments are temporarily deprecated.
                                   
        """
        self.results = []
        layers = list(sorted(layers))
        cc = Constants
        for lang in dataset_language:
            for feature in features:
                for init_strategy in model_init_strategies:
                    title = dataset_name + "_" + _lang(lang) + "_" + feature + "_task_random=" + str(init_strategy) +\
                            "_grads="  +str(enable_grads) + "_variational=" + str(use_variational) + "_poisoned=" + str(poisoning_ratio)
                    writer = tensorboard.SummaryWriter(os.path.join(cc.LOGGING_DIR, title, "layers={}-{}".format(layers[0], layers[-1])))
                    pipe = Probing_pipeline(writer = writer, device = device,
                                            feature = feature, model_path = cc.MODELS_PATH[dataset_name][str(lang)], 
                                            lang = lang, split = dataset_split)
                    pipe.disable_cache()
                    if from_disk: assert isinstance(prefix_data_path, str)
                    pipe.load_data(from_disk = from_disk, data_path = dataset_name if not from_disk else prefix_data_path,
                                   own_feature_set = own_feature_set, only_custom_features = False)
                    if model2probe == Wav2Vec2Prober:
                        print(pipe.preprocess_data(preprocessing_fn, save_path = save_preprocessed_data,
                                                  target_processing = None, **kwargs))
                    print("The task title:", title)
                    res = pipe.run_probing(model2probe, probing_fn, layers = layers, enable_grads = enable_grads, 
                                           use_variational = use_variational, 
                                           plotting_fn = plotting_fn, 
                                           save_checkpoints = save_checkpoints, 
                                           init_strategy = init_strategy, 
                                           plotting_config = {"title": title,
                                                              "custom_features": list(own_feature_set.keys()) if own_feature_set is not None else "None",
                                                              "metrics": ['f1'], 'save_path': cc.GRAPHS_PATH + str(cc.TODAY)},
                                          poisoning_ratio = poisoning_ratio,
                                          poisoning_mapping = poisoning_mapping)
                    pipe.cleanup()
                    if return_results: self.results.append(res)
        if not cc.DEBUG: clear_output(wait = True) 
        # os.system("rm /root/.cache/huggingface/* -r")
        os.system("rm ./cache/ -r")
    def get_testing_results(self):
        """ A format:
        {
        "config": {dict of plotting config including "title"}
        "data": {"loss": [], "metrics": {}}
        }
        """
        return self.results
    def __repr__(self): return "ez4ence"

def main():
    
    layers = [1, 2, 3, 4, 7, 8, 9, 12, 15, 18]
    print(layers)
    TaskTester(dataset_name = "timit_asr",
           model2probe = Wav2Vec2Prober,
           features = ['sex', 'age_bin'],
           layers = layers,
           preprocessing_fn = prepare_probing_task_timit_2,
           use_variational = True,
           enable_grads = False,
           device = torch.device('cuda'),
           probing_fn = ProberModel,
           save_checkpoints = False,
           poisoning_ratio = 0,
           drop_columns = ['word_detail', 'phonetic_detail'])

if __name__ == "__main__": main()
