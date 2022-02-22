
from lib.linguistic_utils import NLPDatasetProcessor
from lib.tokenizers import T5Processor, BertProcessor
from lib.probers import BertOProber, T5EncoderProber, T5EncoderDecoderProber
from lib.base.processing import Processor
from lib.pipeline import Probing_pipeline
from IPython.display import clear_output
from lib.base.clf import ProberModel
from lib.base.constants import Constants
from lib.base.prober import Prober
from lib.base.utils import _lang, _make_directory_structure
from collections import Callable
from typing import Dict, List, Union 

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class SimpleNLPPipeline(object):
    def __init__(self, model2probe: Prober,
                   dataset_name: str,
                   dataset_type: str,
                   feature: str,
                   layers: Union[Dict, List],
                   tokenizer: Processor,
                   data_column: str = 'text',
                   model_path: str = None,
                   download_data: bool = True,
                   dataset_split: str = "test",
                   data_path: str = None,
                   checkpoint_path: str = None,
                   model_init_strategies: list = [None] + ["full"], 
                   use_mdl: bool = False,
                   device: torch.device = torch.device('cpu'), 
                   probing_fn: torch.nn.Module = ProberModel,
                   enable_grads = False,
                   save_checkpoints: bool = False, 
                   return_results: bool = False,
                   **kwargs):
        cc = Constants
        if model_path is None: model_path = cc.MODELS_PATH[dataset_name]["None"] 

        data_proc = NLPDatasetProcessor(dataset_type = dataset_type, 
                                        model_path = model_path, 
                                        filepath = data_path, 
                                        dataset_name = dataset_name, 
                                        feature_column = feature, tokenizer = tokenizer)
        
        data_proc.download_data(download = download_data)

        data_proc.process_dataset(data_col = data_column)           
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
                                    lang = None, split = dataset_split)
            pipe.disable_cache()
            if not download_data: assert isinstance(data_path, str)
            pipe.load_data(from_disk = True, data_path = dataset_name, 
                            own_feature_set = data_proc.tok2label, only_custom_features = False)

            print("The task title:", title)
            print(f"The features are: {data_proc.tok2label}")
            res = pipe.run_probing(model2probe, probing_fn, layers = layers, enable_grads = enable_grads, 
                                use_variational = use_mdl, 
                                checkpoint_path = checkpoint_path,
                                save_checkpoints = save_checkpoints, 
                                init_strategy = init_strategy, 
                                plotting_config = {"title": title,
                                                    "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))})
            pipe.cleanup()
            if return_results: self.results.append(res)
        if not cc.DEBUG: clear_output(wait = True) 



def main():
    # SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
    #                    model_path = None,
    #                    dataset_type = "person",  save_checkpoints = True, download_data = True,
    #                    checkpoint_path = torch.load("t5small.pth").state_dict(),
    #                    feature = 'label', layers = list(np.arange(1, 5, 1)), 
    #                    tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")


    SimpleNLPPipeline(model2probe = T5EncoderDecoderProber, dataset_name = "t5",
                      model_path = None,
                      dataset_type = "person",  save_checkpoints = False, download_data = False,
                    #    checkpoint_path = torch.load("t5small.pth").state_dict(),
                      feature = 'label', layers = {"encoder": list(np.arange(1, 5, 1)), "decoder": list(np.arange(1, 5, 1))}, 
                      tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")




    # SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
    #                    dataset_type = "DiscoEval",  save_checkpoints = False, 
    #                    feature = 'label', layers = list(np.arange(1, 5, 1)), 
    #                    tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")

    # layers = list(np.arange(1, 2, 1))
    # SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "bert", model_path = None,
    #                    dataset_type = "senteval",  save_checkpoints = False, dataset_split = 'train',
    #                    feature = 'label', layers = list(np.arange(1, 2, 1)), 
    #                    tokenizer= BertProcessor, data_path= "past_present.txt", device = torch.device('cuda'), data_column = "data")



if __name__ == "__main__": main()
