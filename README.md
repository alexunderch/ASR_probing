# ASR_probing
A framework for ASR probing using MDL approach
## Installation
```bash
git clone https://github.com/alexunderch/ASR_probing.git
cd ASR_probing
pip install -r requirements.txt
```
## The first thing to know
The main object of the framework is the `Probing_pipeline` class instance:
```
from lib.pipeline import Probing_pipeline
pipe = Probing_pipeline(writer: torch.utils.tensorboard.SummaryWriter, device: torch.device,
                        feature: str, model_path: str, data: Dataset = None, lang: str = None, split: str = None)
```
This class is responsible for preprocessing and whole probing procedure: it has two nice methods: `load_data()` and `run_probing()` which are key for probing. 

## Smth in addition

1. Datasets prerequisites: each probing dataset should be wrapped in HuggingFace `dataset` class in `torh` format with the following features: |`input_values`, `attention_masks`, `label`| -- the first 2 are from tokenizer `transformers` input. 
2. `Probing_pipeline.make_probing` has an instance of `Prober` class as input, this is the wrapper to probing classifier.
3. Hahaha  
