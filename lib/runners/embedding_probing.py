from collections import Callable
from typing import Union, List, Dict
import os
import torch

from torch.utils import tensorboard
from datasets import Dataset, DatasetDict
from ..base.constants import Constants
from ..func_utils import prepare_probing_task, prepare_probing_task_timit, prepare_probing_task_timit_2, prepare_probing_task_
from ..base.utils import _lang, _make_directory_structure, DatasetSplit, f_set
from ..base.processing import Processor
from ..phoneme_utils import comparison_dict
from ..phoneme_utils import *
from ..base.task import TaskTester
from ..clf import ProberModel
from ..probers import Prober, StackedEmbeddingsProber
from ..pipeline import Probing_pipeline
from IPython.display import clear_output
from ..tokenizers import Wav2Vec2OProcessor

class SimpleStackedEmbeddingPipeline(TaskTester):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, embedding_tensor_dataset: Union[Dataset, DatasetDict], 
                        features: Union[List, str],
                        task_name: str,
                        data_path: str = None,
                        dataset_split = "all",
                        model_init_strategies: list = [None] + ["full"], 
                        use_mdl: bool = False,
                        checkpoint: Union[str, Dict] = None,
                        device: torch.device = torch.device('cpu'), 
                        probing_fn: torch.nn.Module = ProberModel,
                        own_feature_set = None,
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

                    writer = tensorboard.SummaryWriter(os.path.join(cc.LOGGING_DIR, title, "layers={}-{}".format(task_name)))
                    pipe = Probing_pipeline(writer = writer, device = device,
                                            feature = feature, data = embedding_tensor_dataset,
                                            lang = None,  split = DatasetSplit(dataset_split))

                    pipe.f_set = f_set(set(embedding_tensor_dataset[feature])) if own_feature_set is None else own_feature_set                        
                    pipe.disable_cache()
                    pipe.load_data(from_disk = True, data_path = data_path)                   

                    print("The task title:", title)
                    res = pipe.run_probing(StackedEmbeddingsProber, probing_fn, enable_grads = False, 
                                use_variational = use_mdl, 
                                checkpoint = checkpoint,
                                save_checkpoints = save_checkpoints, 
                                init_strategy = init_strategy, 
                                plotting_config = {"title": title,
                                                    "metrics": ['f1'], 'save_path': os.path.join(cc.GRAPHS_PATH, str(cc.TODAY))},
                                        poisoning_ratio = poisoning_ratio,
                                        poisoning_mapping = poisoning_mapping)
                    pipe.cleanup()
                    self.results.append(res)
            if not cc.DEBUG: clear_output(wait = True) 
            return self.results
