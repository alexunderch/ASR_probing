from ast import Str
from typing import Dict, Union
import torch
from transformers import Wav2Vec2ForCTC, BertModel, T5ForConditionalGeneration, T5EncoderModel
from datasets import Dataset
from .base.utils import print_if_debug
from .base.constants import Constants
from .base.trainer import Trainer, F1Score
from .base.clf import LinearModel, Loss

from .base.prober import Prober
import os
import numpy as np
from tqdm.autonotebook import tqdm
from copy import deepcopy
from collections import Callable

class BertOProber(Prober):
    def __init__(self, model_path: Union[Str, Dict], writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:
        super().__init__(BertModel, model_path, writer, data, device, init_strategy)
        if init_strategy is not None:
            print_if_debug("reseting network parameters...", self.cc.DEBUG)
            assert isinstance(init_strategy, str)
            if init_strategy == "full": self.model.base_model.init_weights()
            else: print("No init with {} strategy".format(init_strategy))
        self.writer = writer
        self.data = data
        self.device = device
    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, layers: list = [1], from_memory = None, save_outputs: bool = False, task_title: dict = None) -> dict:
        self.fixed_encoder = list(self.model.cpu().encoder.layer)
        self.model.pooler = torch.nn.Identity(1024)
        assert np.alltrue([l > 0 and l < len(self.fixed_encoder) for l in layers])

        def _prepare_data(batch):
            """Helper function
            """
            labels = batch['label'].to(self.device)
            inp_values, att_masks =  batch['input_values'][0].to(self.device), batch['attention_mask'][0].to(self.device)
            return inp_values, att_masks, labels

        print_if_debug("stacking classifiers...", self.cc.DEBUG)

        loss_fn = Loss(use_variational)

        probing_info = {'loss': [], 'metrics': []}

        inputs, attention_masks, _ = _prepare_data(iter(self.dataloader).next())

        model_config = {'in_size': self.model.config.hidden_size * self.cc.POOLING_TO, 
                        'hidden_size': 100,
                        'out_size': len(torch.unique(self.data['label'])),
                        'variational': use_variational,
                        'device': self.device}

        for layer in tqdm(layers, total = len(layers)):
            self.logger.log_string(f"layer {layer} of {len(layers)} in process")     
            self.model.encoder.layer = deepcopy(torch.nn.ModuleList(self.fixed_encoder[:layer]).to(self.device))  

            if enable_grads:   
                for module in self.model.encoder.layer:
                    for param in module.parameters(): param.requires_grad = False

            probing_model = prober(parent_model = self.model,
                                    clf = LinearModel(**model_config),
                                    enable_grads = enable_grads
                                    ).to(self.device)
            probing_model.eval()
            self.writer.add_graph(probing_model, input_to_model = [inputs, attention_masks])
            tr = Trainer(model = probing_model.to(self.device), logger = self.logger, profiler = self.profiler, writer = self.writer,
                         loss_function = loss_fn, optimizer = torch.optim.Adam, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, device = self.device, lr = 3. * 1e-3)
            print_if_debug("training...", self.cc.DEBUG)
            _ = tr.train(train_loader = self.dataloader, batch_processing_fn = _prepare_data, count_of_epoch = self.cc.N_EPOCHS, info = {"layer": layer})
            

            if save_outputs:
                chkpnt = self.checkpointer(probing_model = probing_model.cpu(),
                                            task_title = "" if task_title is None else task_title['title'],
                                            params = model_config, layer_idx = layer, optimizer = tr.optimizer)
                print_if_debug("checkpoint {} saved...".format(chkpnt), self.cc.DEBUG)
            
            self._clear_cache()
            print_if_debug("validating...", self.cc.DEBUG)
            valid_loss, valid_metrics = tr.validate(valid_loader = self.validloader, batch_processing_fn = _prepare_data, metrics = F1Score(task_title["metrics"]), info = {"layer": layer})            
            probing_info['loss'].append(valid_loss)
            probing_info['metrics'].append(valid_metrics) 
            
            probing_model = probing_model.cpu()
            del probing_model
            self.logger.log_string(self.profiler.rep + "\n")

        print_if_debug('running probes...', self.cc.DEBUG)
        return probing_info

class Wav2Vec2Prober(Prober):
    def __init__(self, model_path: str, writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:

        super().__init__(Wav2Vec2ForCTC, model_path, writer, data, device, init_strategy)

        if init_strategy is not None:
            print_if_debug("reseting network parameters...", self.cc.DEBUG)
            assert isinstance(init_strategy, str)
            if init_strategy == "full": self.model.init_weights()
            elif init_strategy == "encoder":
                for p in self.model.wav2vec2.encoder.parameters(): torch.nn.init.normal_(p)
            elif init_strategy == "feature_extractors": 
                for p in self.model.wav2vec2.feature_extractor.parameters(): torch.nn.init.normal_(p)
                for p in self.model.wav2vec2.feature_projection.parameters(): torch.nn.init.normal_(p)
            else: print("No init with {} strategy".format(init_strategy))
        #debugging tools
        self.writer = writer
        self.data = data
        self.device = device

    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, layers: list = [1], from_memory = None, save_outputs: bool = False, task_title: dict = None) -> dict:
        self.model.freeze_feature_extractor()
        self.fixed_encoder = self.model.wav2vec2.encoder.layers.cpu()
        assert np.alltrue([l > 0 and l < len(self.fixed_encoder) for l in layers])

        def make_hidden_states(example, model = self.model, device: torch.device = torch.device('cpu')) -> list:
            """Returns outputs of all model layers 
               Agrs:
                  enable_grads, bool: optional flag, whether to propagate grads or not
                                      default = False
            """
            with torch.no_grad(): 
                output = model(example[0].to(self.device), 
                               attention_mask = example[1].to(self.device),
                               output_hidden_states = True)
            return [hs.cpu().view((len(hs), -1)).numpy() for hs in output.hidden_states]
        
        def _prepare_data(batch):
            """Helper function
            """
            labels = batch['label'].to(self.device)
            inp_values, att_masks =  batch['input_values'][0].to(self.device), batch['attention_mask'][0].to(self.device)
            #getting ready tp encoder
            with torch.no_grad(): 
                extract_features = self.model.wav2vec2.feature_extractor(inp_values).transpose(1, 2)
                hidden_states, extract_features = self.model.wav2vec2.feature_projection(extract_features)
                att_masks = self.model.wav2vec2._get_feature_vector_attention_mask(extract_features.shape[1], att_masks)
                hidden_states = self.model.wav2vec2._mask_hidden_states(hidden_states, attention_mask = att_masks)

            return hidden_states, att_masks, labels

        if from_memory is not None:
            assert isinstance(from_memory, str)
            raise NotImplementedError("") 
        else:
            print_if_debug("stacking classifiers...", self.cc.DEBUG)

            loss_fn = Loss(use_variational)

            probing_info = {'loss': [], 'metrics': []}

            inputs, attention_masks, _ = _prepare_data(iter(self.dataloader).next())

            
            model_config = {'in_size': self.model.config.hidden_size * self.cc.POOLING_TO, 
                            'hidden_size': 100,
                            'out_size': len(torch.unique(self.data['label'])),
                            'variational': use_variational,
                            'device': self.device}

            for layer in tqdm(layers, total = len(layers)):
                self.logger.log_string(f"layer {layer} of {len(layers)} in process")     

                self.model.wav2vec2.encoder.layers = deepcopy(self.fixed_encoder[:layer].to(self.device))     
                if enable_grads:
                    for module in self.model.wav2vec2.encoder.layers[:-1]:
                        for param in module.parameters(): param.requires_grad = False

                probing_model = prober(parent_model = self.model.wav2vec2.encoder,
                                       clf = LinearModel(**model_config),
                                       enable_grads = enable_grads
                                       ).to(self.device)
                                       
                probing_model.eval()
                self.writer.add_graph(probing_model, input_to_model = [inputs, attention_masks])
                tr = Trainer(model = probing_model.to(self.device), logger = self.logger, profiler = self.profiler, writer = self.writer,
                         loss_function = loss_fn, optimizer = torch.optim.Adam, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, device = self.device, lr = 3. * 1e-3)
                print_if_debug("training...", self.cc.DEBUG)
                _ = tr.train(train_loader = self.dataloader, batch_processing_fn = _prepare_data, count_of_epoch = self.cc.N_EPOCHS, info = {"layer": layer})
                

                if save_outputs:
                    chkpnt = self.checkpointer(probing_model = probing_model.cpu(),
                                                task_title = "" if task_title is None else task_title['title'],
                                                params = model_config, layer_idx = layer, optimizer = tr.optimizer)
                    print_if_debug("checkpoint {} saved...".format(chkpnt), self.cc.DEBUG)
                
                self._clear_cache()
                print_if_debug("validating...", self.cc.DEBUG)
                valid_loss, valid_metrics = tr.validate(valid_loader = self.validloader, batch_processing_fn = _prepare_data, metrics = F1Score(task_title["metrics"]), info = {"layer": layer})            
                probing_info['loss'].append(valid_loss)
                probing_info['metrics'].append(valid_metrics)

               
                probing_model = probing_model.cpu()                
                del probing_model
                self.logger.log_string(self.profiler.rep + "\n")

        print_if_debug('running probes...', self.cc.DEBUG)
        return probing_info

class T5Prober(Prober):
    def __init__(self, model_path: str, writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:

        super().__init__(T5EncoderModel, model_path, writer, data, device, init_strategy)
        self.model.config.is_decoder = False

        if init_strategy is not None:
            print_if_debug("reseting network parameters...", self.cc.DEBUG)
            assert isinstance(init_strategy, str)
            if init_strategy == "full": self.model.encoder.init_weights()
            else: print("No init with {} strategy".format(init_strategy))
        #debugging tools
        self.writer = writer
        self.data = data
        self.device = device

    def make_hidden_states(self, example) -> list:
        with torch.no_grad(): 
            output = self.model(example[0].to(self.device), 
                            attention_mask = example[1].to(self.device),
                            output_hidden_states = True,
                            output_attentions = True,
                            return_dict = True)
        return [hs.cpu().view((len(hs), -1)).numpy() for hs in output.hidden_states]
        

    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, layers: list = [1], from_memory = None, save_outputs: bool = False, task_title: dict = None) -> dict:
        self.fixed_encoder = self.model.encoder.block.cpu()

        assert np.alltrue([l > 0 and l < len(self.fixed_encoder) for l in layers])

        def _prepare_data(batch):
            """Helper function
            """
            labels = batch['label'].to(self.device)
            inp_values, att_masks =  batch['input_values'][0].to(self.device), batch['attention_mask'][0].to(self.device)
            return inp_values, att_masks, labels


        print_if_debug("stacking classifiers...", self.cc.DEBUG)

        loss_fn = Loss(use_variational)

        probing_info = {'loss': [], 'metrics': []}

        inputs, attention_masks, _ = _prepare_data(iter(self.dataloader).next())

        
        model_config = {'in_size': self.model.config.hidden_size * self.cc.POOLING_TO, 
                        'hidden_size': 100,
                        'out_size': len(torch.unique(self.data['label'])),
                        'variational': use_variational,
                        'device': self.device}

        for layer in tqdm(layers, total = len(layers)):
            self.logger.log_string(f"layer {layer} of {len(layers)} in process")     

            self.model.encoder.block = deepcopy(torch.nn.ModuleList(self.fixed_encoder[:layer]).to(self.device))       
            if enable_grads:
                for module in list(self.model.encoder.block)[:-1]:
                    for param in module.parameters(): param.requires_grad = False

            probing_model = prober(parent_model = self.model,
                                    clf = LinearModel(**model_config),
                                    enable_grads = enable_grads
                                    ).to(self.device)
            probing_model.eval()
            self.writer.add_graph(probing_model, input_to_model = [inputs, attention_masks])
            tr = Trainer(model = probing_model.to(self.device), logger = self.logger, profiler = self.profiler, writer = self.writer,
                    loss_function = loss_fn, optimizer = torch.optim.Adam, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, device = self.device, lr = 3. * 1e-3)
            print_if_debug("training...", self.cc.DEBUG)
            _ = tr.train(train_loader = self.dataloader, batch_processing_fn = _prepare_data, count_of_epoch = self.cc.N_EPOCHS, info = {"layer": layer})
            

            if save_outputs:
                chkpnt = self.checkpointer(probing_model = probing_model.cpu(),
                                            task_title = "" if task_title is None else task_title['title'],
                                            params = model_config, layer_idx = layer, optimizer = tr.optimizer)
                print_if_debug("checkpoint {} saved...".format(chkpnt), self.cc.DEBUG)
            
            self._clear_cache()
            print_if_debug("validating...", self.cc.DEBUG)
            valid_loss, valid_metrics = tr.validate(valid_loader = self.validloader, batch_processing_fn = _prepare_data, metrics = F1Score(task_title["metrics"]), info = {"layer": layer})            
            probing_info['loss'].append(valid_loss)
            probing_info['metrics'].append(valid_metrics)   

            probing_model = probing_model.cpu()
            del probing_model
            self.logger.log_string(self.profiler.rep + "\n")

        print_if_debug('running probes...', self.cc.DEBUG)
        return probing_info
