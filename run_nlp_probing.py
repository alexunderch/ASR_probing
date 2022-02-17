from lib.linguistic_utils import NLPDatasetProcessor
from lib.tokenizers import T5Processor, BertProcessor
from lib.probers import BertOProber, T5Prober
from lib.base.processing import Processor
from lib.pipeline import Probing_pipeline, _make_directory_structure
from IPython.display import clear_output
from lib.base.clf import ProberModel
from lib.base.constants import Constants
from lib.base.prober import Prober
from lib.base.utils import _lang
from collections import Callable
import os
import torch
from torch.utils import tensorboard


def main():
    class SimpleNLPPipeline(object):
        def __init__(self, model2probe: Prober,
                       dataset_name: str,
                       dataset_type: str,
                       feature: str,
                       layers: list,
                       tokenizer: Processor,
                       data_column: str = 'text',
                       download_data: bool = True,
                       dataset_language: list = [None],
                       dataset_split: str = "test",
                       data_path: str = None,
                       checkpoint_path: str = None,
                       model_init_strategies: list = [None] + ["full"], 
                       use_mdl: bool = False,
                       device: torch.device = torch.device('cpu'), 
                       probing_fn: torch.nn.Module = ProberModel,
                       enable_grads = False,
                       plotting_fn: Callable = None,
                       save_checkpoints: bool = False, 
                       return_results: bool = False,
                       **kwargs):

            data_proc = NLPDatasetProcessor(dataset_type = dataset_type, model_path = cc.MODELS_PATH[dataset_name][str(lang)], filepath = data_path, 
                                    dataset_name = dataset_name, feature_column = feature, tokenizer = tokenizer)
            data_proc.download_data(download = download_data)

            data_proc.process_dataset(data_col = data_column, load_from_disk = not download_data)           
            self.results = []
            layers = list(sorted(layers))

            cc = Constants
            _ = _make_directory_structure()
            for lang in dataset_language:
                for init_strategy in model_init_strategies:
                    title = dataset_name + "_" + _lang(lang) + "_" + feature + "_task_random=" + str(init_strategy) +\
                            "_grads="  +str(enable_grads) + "_variational=" + str(use_variational)
                    writer = tensorboard.SummaryWriter(os.path.join(cc.LOGGING_DIR, title, "layers={}-{}".format(layers[0], layers[-1])))
                    pipe = Probing_pipeline(writer = writer, device = device,
                                            feature = feature, model_path = cc.MODELS_PATH[dataset_name][str(lang)], 
                                            lang = lang, split = dataset_split)
                    pipe.disable_cache()
                    if from_disk: assert isinstance(data_path, str)
                    pipe.load_data(from_disk = from_disk, data_path = dataset_name if download_data else data_path,
                                own_feature_set = data_proc.tok2label, only_custom_features = False)
                    
                    print("The task title:", title)
                    res = pipe.run_probing(model2probe, probing_fn, layers = layers, enable_grads = enable_grads, 
                                        use_variational = use_mdl, 
                                        plotting_fn = None, 
                                        save_checkpoints = save_checkpoints, 
                                        init_strategy = init_strategy, 
                                        plotting_config = {"title": title,
                                                            "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))})
                    pipe.cleanup()
                    if return_results: self.results.append(res)
            if not cc.DEBUG: clear_output(wait = True) 
    

    
if __name__ == "__main__": main()
