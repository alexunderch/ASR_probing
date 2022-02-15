#NLP probing algorithm
##For "hard-coded" datasets
The full pipepline of working with these datasets consists of 2 great stages: dataset processing and running a chosen task which are combined in `SimpleNLPPipeline`.
You also feel free to use own beautiful pipelines.

The pipeline can be instantiated in the following way:
```python
from run_nlp_probing import SimpleNLPPipeline
task = SimpleNLPPipeline(...)
```
Yes, it is that simple. Let us to figure out the input parameters:
Dataset preprocessing parameters:
- `dataset_type`(`str`): the name of a hardcoded dataset. The following ones are available: 'senteval', 'person', 'DiscoEval'('DC', 'SP', 'PDTB') 
(others are in process)
- `dataset_split`(`str`): the dataset split which would be used for probing (default = 'test')
- `filepath` (`str`): a filepath to downloaded dataset files (for 'senteval' it should be like filename.txt); 
Note: for 'DiscoEval' is should be ('DC', 'SP', 'PDTB') -- the name of used set.
- `dataset_name`(`str`): a name the dataset should be saved under.
- `feature_column` (`str`):
- `tokenizer` (`lib.base.Processor`):
- `download` (`bool`):
- `data_column` (`str`):
Dataset preprocessing parameters:
- `model_path` (`str`):
- `checkpoint_path` (`str`):         
- `model_init_strategies` (`list`): = [None] + ["full"], 
- `use_variational` (`bool`): = False,
- `device` (`torch.device`): = torch.device('cpu'), 
- `probing_fn` (`torch.nn.Module`): = ProberModel,
- `enable_grads`(`bool`): = False,
- `save_checkpoints` (`bool`): = False, 
- `save_preprocessed_data` (`str`): = None,
- `return_results` (`bool`): = False,

