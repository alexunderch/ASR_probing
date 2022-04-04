from collections import Callable
from typing import Optional, Union, List, Dict
import os
import torch

from torch.utils import tensorboard
from datasets import Dataset, DatasetDict
from ..base.constants import Constants
from ..base.utils import _lang, _make_directory_structure, DatasetSplit, f_set
from ..base.processing import Processor
from ..base.task import TaskTester
from ..clf import ProberModel
from ..tokenizers import EmbeddingProcessor
from ..probers import Prober, StackedEmbeddingsProber
from ..utils import EmbeddingTensorDataset
from ..pipeline import Probing_pipeline
from IPython.display import clear_output

class SimpleStackedEmbeddingPipeline(TaskTester):
    def __init__(self, embedding_tensor_dataset: Union[Dataset, DatasetDict], 
                        features: Union[List, str],
                        task_name: str,
                        data_path: str = None,
                        dataset_split = "all",
                        data_column:str = "data",
                        model_init_strategies: list = [None] + ["full"], 
                        use_mdl: bool = False,
                        checkpoint: Union[str, Dict] = None,
                        device: torch.device = torch.device('cpu'), 
                        probing_fn: torch.nn.Module = ProberModel,
                        own_feature_set: dict = None,
                        tokenizer: Optional[Callable] = EmbeddingProcessor,
                        save_checkpoints: bool = False, 
                        poisoning_ratio: float = 0,
                        poisoning_mapping = None, **kwargs) -> dict:
            self.results = []
            cc = Constants    
            _ = _make_directory_structure()
            for feature in features:                    
                for init_strategy in model_init_strategies:
                    title = task_name + "_" + feature + "_task_random=" + str(init_strategy) +\
                            "_grads="  +str(False) + "_variational=" + str(use_mdl) + (("_poisoned=" + str(poisoning_ratio)) if poisoning_ratio > 0 else str())

                    writer = tensorboard.SummaryWriter(os.path.join(cc.LOGGING_DIR, title, "task={}".format(task_name)))
                    pipe = Probing_pipeline(writer = writer, device = device, model_path= None,
                                            feature = feature, data = embedding_tensor_dataset,
                                            lang = None,  split = DatasetSplit(dataset_split))

                    pipe.disable_cache()
                    pipe.load_data(from_disk = (data_path is not None), data_path = data_path)  
                    pipe.f_set = f_set(set(pipe.dataset[feature])) if own_feature_set is None else own_feature_set                        

                    data_proc = EmbeddingTensorDataset(feature_column=feature, tokenizer=tokenizer, dataset=pipe.dataset,
                                                       f_set = own_feature_set, only_custom_features = True)
                                  
                    pipe.dataset = data_proc.process_dataset(preprocessing_fn = None, data_column = data_column)
                    print("The task title:", title)
                    res = pipe.run_probing(StackedEmbeddingsProber, probing_fn, enable_grads = False, 
                                use_variational = use_mdl, 
                                checkpoint = checkpoint,
                                layers=[],
                                save_checkpoints = save_checkpoints, 
                                init_strategy = init_strategy, 
                                plotting_config = {"title": title,
                                                    "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))},
                                        poisoning_ratio = poisoning_ratio,
                                        poisoning_mapping = poisoning_mapping)
                    pipe.cleanup()
                    self.results.append(res)
            if not cc.DEBUG: clear_output(wait = True) 
