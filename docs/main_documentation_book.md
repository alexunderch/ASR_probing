# Probing framework

One of the interpretation tools we could use for deep learning models is probing. Here we present a framework to probe various types of models from HuggingFace [transformers](https://huggingface.co/docs/transformers/index) library.

The key feature of our framework is _MDL_ (minimum descriprion length) approach:

## MDL
Our approach is based on __crude MDL__: for each probing task $t \in \mathcal{T}$ we denote probing model $\mathcal{M}$ and a probing classifier $\mathcal{C}$ such that for input ($\mathcal{X_t}$) and output spaces ($\mathcal{Y_t}$): 
\begin{align}
\forall x \in \mathcal{X_t}, \ z = sg[\mathcal(M)](x) \in \mathcal{Z_t}, \ y = \mathcal{C}(z), \\
\min_{\theta} \{ KL(p^{\mathcal{C}}_{\theta} || p^{\mathcal{M}}) - \sum_{y \backsim p^{\mathcal{C}}} y \log p^{\mathcal{C}}_{\theta} \}
\end{align}

Here, $KL$ is a regularization term tweaking a classifier distribution towards the probed model one. It's done to not overfit to training data and to use a model's knowledge too.

## Probing cases with examples
### NLP Probing (For "hard-coded" datasets)

The full pipepline of working with these datasets consists of 2 great stages: dataset processing and running a chosen task which are combined in `SimpleNLPPipeline`.
You also feel free to use own beautiful pipelines.

The pipeline can be instantiated in the following way:



```python
from lib.runners.nlp_probing import *
task = SimpleNLPPipeline(...)
```


Yes, it is that simple. Let us to figure out the input parameters:\
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
- `data_column` (`str`): a column with text to tokenize (`defult = 'data'`, actual only for CSV/TSV files)
- `data_path` (`str`): a path to save the preprocessed probing dataset 
- `is_prepared` (`bool`): to preprocess downloaded dataset or not (`default=False`, active only if `download_data=True`)\
Model configuration parameters:
- `model2probe` (`lib.base.Prober`): a type of prober model (a wrapper for HuggingFace model for convenience) to use. Look more at the [file](https://github.com/alexunderch/ASR_probing/blob/main/lib/probers.py) for all details.
- `layers (Union[List, Dict])`: layers of a model to probe on (should be less than layers in the model, `default =[]`)
- `model_path` (`str`): a name of model to use ('bert', 't5' for nlp tasks); An important note: if `dataset_name` if not 't5' or 'bert' (then models load automatically), you should set this argument like 't5-small', 'bert-small-cased' etc.
- `checkpoint` (`str` or `.state_dict()`): a path to checkpoint for standard `torch.nn.Module`  models or state dict for HuggingFaceModels: you can specify checkpoint in three ways: if the file is availabe and suitable for the used model, it can be instantiated like str-filepath, or you can specify str-address for huggingface hub; if it is available but not suitable, you can prepare statedict for the model and specify it. (`default = None`)        
- `model_init_strategies` (`list`): a list of model initialized stategies (`default =  [None, "full"]`)
- `use_mdl` (`bool`): use MDL (variational approach) or not (`default = False`)
- `device` (`torch.device`): device to probe on (`default = torch.device('cpu')`) 
- `probing_fn` (`torch.nn.Module`): a class of prbong classier (`default = ProberModel`)
- `enable_grads`(`bool`): a flag backprop through last layers or not (`default = False`)
- `save_checkpoints` (`bool`): save checkpoints for each layer or not (`default = False`) 
- `return_results` (`bool`): (`default = False`)


Probing the first 4 layers of `t5-small` model using prepared checkpoint on [person senteval task](https://github.com/facebookresearch/SentEval/tree/main/data/probing)


```python
SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
                   model_path = None,
                   dataset_type = "person",  save_checkpoints = True, download_data = True,
                   checkpoint_path = torch.load("t5small.pth").state_dict(),
                   feature = 'label', layers = list(np.arange(1, 5, 1)), 
                   tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), 
                   data_column = "text")
```

using encoder-decoder architecture:


```python
SimpleNLPPipeline(model2probe = T5EncoderDecoderProber, dataset_name = "t5",
                  model_path = None,
                  dataset_type = "person",  save_checkpoints = False, download_data = False,
                  feature = 'label', 
                  layers = {"encoder": list(np.arange(1, 5, 1)), "decoder": list(np.arange(1, 5, 1))}, 
                  tokenizer= T5Processor, data_path= "person.tsv", 
                  device = torch.device('cuda'), data_column = "text"
```

`BERT` with a custom dataset name and, saving checkpoints on [discoeval](https://github.com/ZeweiChu/DiscoEval) dataset


```python

SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                   morph_call = "DiscoEval",  save_checkpoints = False, 
                   feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "all",
                   tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")

SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                   checkpoint = "bert-base-cased",
                   morph_call = "DiscoEval",  save_checkpoints = True, 
                   feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "dev;test", 
                   download_data = True,
                   tokenizer= BertProcessor, data_path= "DC", device = torch.device('cuda'), data_column = "data")

```

### ASR Probing (For "hard-coded" datasets)
The full pipepline of working with these datasets consists of 2 great stages: dataset processing and running a chosen task which are combined in `SimpleASRPipeline`.
You also feel free to use own beautiful pipelines.

The pipeline can be instantiated in the following way:


```python
from lib.runners.asr_probing import *
task = SimpleASRPipeline(...)
```

Yes, it is that simple. Let us to figure out the input parameters:\
Dataset preprocessing parameters:
- `morph_call`(`str`): the name of a hardcoded dataset prepared for probing task. The following ones are available: 'common_voice', 'timit_asr'
(others are in process)
- `dataset_split`(`str`): the dataset split which would be used for probing; huggingface splits are available [link](https://huggingface.co/docs/datasets/splits.html) but you can also use composed splits (e.g. 'dev;test' -- separated by ";") and use "all"  to concatenate all splits in one dataset (`default = 'test'`)
- `filepath` (`str`): a filepath to downloaded dataset files (for 'senteval' it should be like filename.txt); 
- `dataset_name`(`str`): a name the dataset should be saved under.
- `dataset_language` (`List[str]`): list of languages for `morph_call='common_voice'`
- `features` (`Union[List, str]`): a column which should be probe on (should be in the probing dataset)
- `tokenizer` (`lib.base.Processor`): a tokenizer class. 
- `data_path` (`str`): a path to save the preprocessed probing dataset 
- `own_feature_set` (`dict)`: use own features (`dict(str: int)`) `default = None`
- `from_disk` (`bool`): take data from own disk or from the HuggingFace Hub (`default=False`)
Model configuration parameters:
- `model2probe` (`lib.base.Prober`): a type of prober model (a wrapper for HuggingFace model for convenience) to use. Look more at the [file](https://github.com/alexunderch/ASR_probing/blob/main/lib/probers.py) for all details.
- `layers (Union[List, Dict])`: layers of a model to probe on (should be less than layers in the model, `default =[]`)
- `model_path` (`str`): a name of model to use ('wav2vec2' for asr tasks) or set `None` but set `morph_call`
- `checkpoint` (`str` or `.state_dict()`): a path to checkpoint for standard `torch.nn.Module`  models or state dict for HuggingFaceModels: you can specify checkpoint in three ways: if the file is availabe and suitable for the used model, it can be instantiated like str-filepath, or you can specify str-address for huggingface hub; if it is available but not suitable, you can prepare statedict for the model and specify it. (`default = None`)     
- `preprocessing_fn: Callable`: a funtion to read audio data from files, should return batch with 'len_speech', 'specch', 'sampling_rate' (look [there](https://github.com/alexunderch/ASR_probing/blob/main/lib/func_utils.py) for details, `default=prepare_probing_task`)
- `model_init_strategies` (`list`): a list of model initialized stategies (`default =  [None, "full"]`)
- `use_mdl` (`bool`): use MDL (variational approach) or not (`default = False`)
- `device` (`torch.device`): device to probe on (`default = torch.device('cpu')`) 
- `probing_fn` (`torch.nn.Module`): a class of prbong classier (`default = ProberModel`)
- `enable_grads`(`bool`): a flag backprop through last layers or not (`default = False`)
- `save_checkpoints` (`bool`): save checkpoints for each layer or not (`default = False`) 
- `use_ctc_objectve`(`bool`): use standard [Cross-Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) for classification of [CTC loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) (`default=False`)
- `return_results` (`bool`): (`default = False`)


`gender` classification with `common voice` dataset using `Wav2Vec2`


```python
SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "common_voice", model_path = None,
                    save_checkpoints = False, dataset_language = ["ru"], dataset_name = None,
                    preprocessing_fn=prepare_probing_task_,
                   features = ['gender'], layers = list(np.arange(1, 3, 1)),  dataset_split= "test",
                   tokenizer= Wav2Vec2OProcessor, data_path= None, device = torch.device('cuda'), 
                   data_column = "data", revision = "1.10.0")
    
```

with own feature set from disk


```python
SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "timit_asr", model_path = None,
                    save_checkpoints = False, dataset_language = [None], dataset_name = "timit_asr_all_consonants",
                    preprocessing_fn=None, from_disk = True, use_mdl = True,
                   features = ['ipa'], layers = list(np.arange(1, 25, 1)),  dataset_split= None,
                   own_feature_set = comparison_dict(ipa_all),
                   tokenizer= Wav2Vec2OProcessor, data_path= "./phonetic_set", device = torch.device('cuda'), 
                   data_column = "speech")

SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "timit_asr", model_path = None,
                save_checkpoints = False, dataset_language = [None], dataset_name = "timit_asr_all_consonants",
                preprocessing_fn=None, from_disk = True, use_mdl = False,
                features = ['ipa'], layers = list(np.arange(1, 25, 1)),  dataset_split= None, 
                own_feature_set = comparison_dict(ipa_all),
                tokenizer= Wav2Vec2OProcessor, data_path= "./phonetic_set", device = torch.device('cuda'), 
                data_column = "speech") 
```

using `ctc`


```python
for use_mdl in [False, True]:
    SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "timit_asr", model_path = None,
                    save_checkpoints = False, dataset_language = [None], dataset_name = None,
                    preprocessing_fn=prepare_probing_task_timit, from_disk = False, 
                    features = ['text'], layers = list(np.arange(1, 25, 1)),  dataset_split= "test",
                    use_mdl = use_mdl, use_ctc_objective = True,
                    own_feature_set = comparison_dict([list(ipa_hardcoded.intersection(ipa_vowels)),
                    list(ipa_hardcoded.intersection(ipa_consonants))], True),
                    tokenizer= Wav2Vec2PProcessor, data_path= None, device = torch.device('cuda'), 
                    data_column = "speech") 
```

### Not stable: own-formatted pipelines

To create your own pipeline, please follow these steps:
* Inherit from `lib.base.Processor` class and override `__call__` method
* Inherit from `lib.base.DatasetProcessor` class and override `preprocess_dataset` method
* Inherit from `lib.base.Prober` class and override `__init__` and `make_probing` methods

An example (TBD):


```python

```
