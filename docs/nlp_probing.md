# NLP probing algorithm
## For "hard-coded" datasets
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
- `feature_column` (`str`): a column which should be probe on (should be in the probing dataset)
- `tokenizer` (`lib.base.Processor`): a tokenizer class. 
- `download_data` (`bool`): whether to use proper the first time or not (default = True)
- `data_column` (`str`): a column with text to tokenize (defult = 'data', actual only for CSV/TSV files)
Dataset preprocessing parameters:
- `model_path` (`str`): a name of model to use ('bert', 't5' for nlp tasks); An important note: if `dataset_name` if not 't5' or 'bert' (then models load automatically), you should set this argument like 't5-small', 'bert-small-cased' etc.
- `checkpoint_path` (`str`): a path to checkpoint for standard `torch.nn.Module`  models or state dict for HuggingFaceModels (default = None)        
- `model_init_strategies` (`list`): a list of model initialized stategies (default =  [None] + ["full"])
- `use_mdl` (`bool`): use MDL (variational approach) or not (default = False)
- `device` (`torch.device`): device to probe on (default = torch.device('cpu')) 
- `probing_fn` (`torch.nn.Module`): a class of prbong classier (default = ProberModel)
- `enable_grads`(`bool`): a flag backprop through or not (default = False)
- `save_checkpoints` (`bool`): save checkpoints for each layer or not (default = False) 
- `return_results` (`bool`): (default = False)

Examples of usage:
  ```python
    SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
                       model_path = None,
                       dataset_type = "person",  save_checkpoints = True, download_data = True,
                       checkpoint_path = torch.load("t5small.pth").state_dict(),
                       feature = 'label', layers = list(np.arange(1, 5, 1)), 
                       tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")

    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                       dataset_type = "DiscoEval",  save_checkpoints = False, 
                       feature = 'label', layers = list(np.arange(1, 5, 1)), 
                       tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")

    
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "bert", model_path = None,
                       dataset_type = "senteval",  save_checkpoints = False, dataset_split = 'train',
                       feature = 'label', layers = list(np.arange(1, 5, 1)), 
                       tokenizer= BertProcessor, data_path= "past_present.txt", 
                       device = torch.device('cuda'), data_column = "data")
  ```
