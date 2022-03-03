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
- `morph_call`(`str`): the name of a hardcoded dataset prepared for probing task. The following ones are available: 'senteval', 'person', 'DiscoEval'('DC', 'SP', 'PDTB') 
(others are in process)
- `dataset_split`(`str`): the dataset split which would be used for probing; huggingface splits are available [link](https://huggingface.co/docs/datasets/splits.html) but you can also use composed splits (e.g. 'dev;test' -- separated by ";") and use "all"  to concatenate all splits in one dataset (`default = 'test'`)
- `filepath` (`str`): a filepath to downloaded dataset files (for 'senteval' it should be like filename.txt); 
Note: for 'DiscoEval' is should be ('DC', 'SP', 'PDTB') -- the name of used set.
- `dataset_name`(`str`): a name the dataset should be saved under.
- `feature_column` (`str`): a column which should be probe on (should be in the probing dataset)
- `tokenizer` (`lib.base.Processor`): a tokenizer class. 
- `download_data` (`bool`): whether to use proper the first time or not (`default = True`)
- `data_column` (`str`): a column with text to tokenize (defult = 'data', actual only for CSV/TSV files)
- `data_path` (`str`): a path to save the preprocessed probing dataset \
Dataset preprocessing parameters:
- `model_path` (`str`): a name of model to use ('bert', 't5' for nlp tasks); An important note: if `dataset_name` if not 't5' or 'bert' (then models load automatically), you should set this argument like 't5-small', 'bert-small-cased' etc.
- `checkpoint` (`str` or `.state_dict()`): a path to checkpoint for standard `torch.nn.Module`  models or state dict for HuggingFaceModels: you can specify checkpoint in three ways: if the file is availabe and suitable for the used model, it can be instantiated like str-filepath, or you can specify str-address for huggingface hub; if it is available but not suitable, you can prepare statedict for the model and specify it. (`default = None`)        
- `model_init_strategies` (`list`): a list of model initialized stategies (`default =  [None, "full"]`)
- `use_mdl` (`bool`): use MDL (variational approach) or not (`default = False`)
- `device` (`torch.device`): device to probe on (`default = torch.device('cpu')`) 
- `probing_fn` (`torch.nn.Module`): a class of prbong classier (`default = ProberModel`)
- `enable_grads`(`bool`): a flag backprop through last layers or not (`default = False`)
- `save_checkpoints` (`bool`): save checkpoints for each layer or not (`default = False`) 
- `return_results` (`bool`): (`default = False`)

Examples of usage:
  ```python
    SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
                       model_path = None,
                       morph_call = "person",  save_checkpoints = True, download_data = True,
                       checkpoint_path = torch.load("t5small.pth").state_dict(),
                       feature = 'label', layers = list(np.arange(1, 5, 1)), 
                       tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")

    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                       morph_call = "DiscoEval",  save_checkpoints = False, 
                       feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "all",
                       tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")
    
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset3", model_path = 'bert-base-cased',
                       checkpoint = "bert-base-cased",
                       morph_call = "DiscoEval",  save_checkpoints = False, 
                       feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "dev;test", download_data = True,
                       tokenizer= BertProcessor, data_path= "DC", device = torch.device('cuda'), data_column = "data")
                       
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "bert", model_path = None,
                       dataset_type = "senteval",  save_checkpoints = False, dataset_split = 'train',
                       feature = 'label', layers = list(np.arange(1, 5, 1)), use_mdl = True,
                       tokenizer= BertProcessor, data_path= "past_present.txt", 
                       device = torch.device('cuda'), data_column = "data")
        
  ```
With encoder-decoder architecture:
```python
    SimpleNLPPipeline(model2probe = T5EncoderDecoderProber, dataset_name = "t5",
                      model_path = None,
                      morph_call = "person",  save_checkpoints = False, download_data = False,
                      checkpoint_path = torch.load("t5small.pth").state_dict(),
                      feature = 'label', layers = {"encoder": list(np.arange(1, 5, 1)), "decoder": list(np.arange(1, 5, 1))}, 
                      tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")


```
  
## For own-formatted datasets (TBD)
For own-formatted dataset it is important to:
1. to inherit from the `lib.base.processing.Processor` class and override `__call__()` method (pass the tests to verity your work);
2. to inherit from the `lib.base.processing.DatasetProcessor` class and override `process_dataset()` method and follow _formatting rules_ (TBD) (pass the tests to verity your work);
3. to pass it through `TaskTester` class (TBD).


## Plotting results
```python
    from plott import plot
    """
    U need 4 experiments: with init strategies None and "full" and with use_mdl = False, True to plot.
    """
    plot(x = layers used in pipeline, experiment_date = 'yyyy-mm-dd', 
         dataset='dataset name used in pipeline', feature = 'label', lang="en", grads='enabled grads in pipeline')
```
