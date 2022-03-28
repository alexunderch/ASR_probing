from typing import Dict, Tuple, Union, List, ByteString
from xml.parsers.expat import model
import torch
from transformers import Wav2Vec2ForCTC, BertModel, T5ForConditionalGeneration, T5EncoderModel
from datasets import Dataset
from .base.utils import print_if_debug
from .base.constants import Constants
from .base.trainer import Trainer, F1Score
from .base.layers import LinearModel, Loss, CTCLikeLoss

from .base.prober import Prober
import os
import numpy as np
from copy import deepcopy
from collections import Callable


from .base.utils import test_ipkernel

if test_ipkernel(): from tqdm.notebook import tqdm 
else: from tqdm import tqdm

class BertOProber(Prober):
    def __init__(self, model_path: Union[str, Dict], writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:
        super().__init__(BertModel, model_path, writer, data, device, init_strategy)
        self.model.config.is_decoder = False
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
        assert np.alltrue([l > 0 and l <= len(self.fixed_encoder) for l in layers])

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
        self.model.config.is_decoder = False
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
    
    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, layers: list = [1], from_memory = None, save_outputs: bool = False, task_title: dict = None, **kwargs) -> dict:
        self.model.freeze_feature_extractor()
        self.fixed_encoder = self.model.wav2vec2.encoder.layers.cpu()
        assert np.alltrue([l > 0 and l <= len(self.fixed_encoder) for l in layers])

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
            labels = batch['label'][0].to(self.device) if self.use_ctc else batch['label'].to(self.device)
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

            loss_fn = Loss(use_variational, ctc = self.use_ctc,  
                           blank_token = self.model.config.pad_token_id)

            probing_info = {'loss': [], 'metrics': []}

            inputs, attention_masks, labels = _prepare_data(iter(self.dataloader).next())
            self.cc.POOLING_TO = labels.size(1) if self.use_ctc else self.cc.POOLING_TO
            model_config = {'in_size': self.model.config.hidden_size * self.cc.POOLING_TO if not self.use_ctc else self.model.config.hidden_size, 
                            'hidden_size': 100,
                            'out_size': self.model.config.vocab_size if self.use_ctc else len(torch.unique(self.data['label'])),
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
                                       enable_grads = enable_grads,
                                       ).to(self.device)
                
                probing_model.use_ctc = self.use_ctc                       
                probing_model.eval()

                self.writer.add_graph(probing_model, input_to_model = [inputs, attention_masks])
                tr = Trainer(model = probing_model.to(self.device), logger = self.logger, profiler = self.profiler, writer = self.writer,
                         loss_function = loss_fn, optimizer = torch.optim.AdamW, scheduler = None, device = self.device, lr = 3 * 1e-3, **kwargs)
                print_if_debug("training...", self.cc.DEBUG)
                _ = tr.train(train_loader = self.dataloader, batch_processing_fn = _prepare_data, count_of_epoch = self.cc.N_EPOCHS, info = {"layer": layer, 'ctc': self.use_ctc})
                

                if save_outputs:
                    chkpnt = self.checkpointer(probing_model = probing_model.cpu(),
                                                task_title = "" if task_title is None else task_title['title'],
                                                params = model_config, layer_idx = layer, optimizer = tr.optimizer)
                    print_if_debug("checkpoint {} saved...".format(chkpnt), self.cc.DEBUG)
                
                self._clear_cache()
                print_if_debug("validating...", self.cc.DEBUG)
                valid_loss, valid_metrics = tr.validate(valid_loader = self.validloader, batch_processing_fn = _prepare_data, 
                                                        metrics = F1Score(task_title["metrics"]), info = {"layer": layer, "ctc": self.use_ctc})            
                probing_info['loss'].append(valid_loss)

                probing_info['metrics'].append(valid_metrics)

               
                probing_model = probing_model.cpu()                
                del probing_model
                self.logger.log_string(self.profiler.rep + "\n")

        print_if_debug('running probes...', self.cc.DEBUG)
        return probing_info


class T5EncoderProber(Prober):
    def __init__(self, model_path: str, writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:

        super().__init__(T5EncoderModel, model_path, writer, data, device, init_strategy)

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


class T5EncoderDecoderProber(Prober):
    def __init__(self, model_path: str, writer: torch.utils.tensorboard.SummaryWriter, data: Dataset = None, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:

        super().__init__(T5ForConditionalGeneration, model_path, writer, data, device, init_strategy)

        if init_strategy is not None:
            print_if_debug("reseting network parameters...", self.cc.DEBUG)
            assert isinstance(init_strategy, str)
            if init_strategy == "full": 
                for module in self.model.modules(): self.model._init_weights(module)
            else: print("No init with {} strategy".format(init_strategy))
        #debugging tools
        self.writer = writer
        self.data = data
        self.device = device

        self.model.fixed_ = None

    def make_hidden_states(self, example) -> list:
        with torch.no_grad(): 
            output = self.model(example[0].to(self.device), 
                            attention_mask = example[1].to(self.device),
                            output_hidden_states = True,
                            output_attentions = True,
                            return_dict = True)
        return [hs.cpu().view((len(hs), -1)).numpy() for hs in output.hidden_states]


    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, layers: dict = {"encoder": [1], "decoder": [1]}, from_memory = None, save_outputs: bool = False, task_title: dict = None) -> dict:
        self.fixed_ = {'encoder': self.model.encoder.block.cpu(), 'decoder': self.model.decoder.block.cpu()}
        self.lm_head = torch.nn.Identity(512)
        assert np.alltrue([l > 0 and l <= len(self.fixed_["encoder"]) for l in layers['encoder']])
        assert np.alltrue([l > 0 and l <= len(self.fixed_["decoder"]) for l in layers['decoder']])

        def _prepare_data(batch):
            """Helper function
            """
            labels = batch['label'].to(self.device)
            inp_values, att_masks =  batch['input_values'][0].to(self.device), batch['attention_mask'][0].to(self.device)
            return inp_values, att_masks, labels


        print_if_debug("stacking classifiers...", self.cc.DEBUG)

        loss_fn = Loss(use_variational)


        inputs, attention_masks, _ = _prepare_data(iter(self.dataloader).next())

        probing_info ={'encoder': {'loss': [], 'metrics': []}, 'decoder': {'loss': [], 'metrics': []}}

        model_config = {'in_size': self.model.config.hidden_size * self.cc.POOLING_TO, 
                        'hidden_size': 100,
                        'out_size': len(torch.unique(self.data['label'])),
                        'variational': use_variational,
                        'device': self.device}
        
        def _probe(model_config: dict, mode: str, layer: int) -> Tuple[float, float]:

            self.logger.log_string(f"layer {layer} of {len(layers[mode])} [{mode}] in process")     
            getattr(self.model, mode).block = deepcopy(torch.nn.ModuleList(self.fixed_[mode][:layer]).to(self.device))       
            if enable_grads:
                for module in list(getattr(self.model, mode).block)[:-1]:
                    for param in module.parameters(): param.requires_grad = False

            probing_model = prober(parent_model = self.model,
                                    clf = LinearModel(**model_config),
                                    enable_grads = enable_grads,
                                    encoder_decoder = False if mode == "encoder" else True,
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

            probing_model = probing_model.cpu()
            del probing_model
            self.logger.log_string(self.profiler.rep + "\n")
            return valid_loss, valid_metrics

        print_if_debug('making probes...', self.cc.DEBUG)
        for m in ['encoder', 'decoder']:    
            for layer in tqdm(layers[m], total = len(layers[m])):
                prinfo = _probe(model_config = model_config, mode = m, layer = layer)
                probing_info[m]['loss'].append(prinfo[0])
                probing_info[m]['metrics'].append(prinfo[1])
                
            

        print_if_debug('done probes...', self.cc.DEBUG)
        return probing_info



###########MAINTENANCE
class StackedEmbeddingsProber(Prober):
    def __init__(self, embeddings: List[torch.TensorType], writer: torch.utils.tensorboard.SummaryWriter, device: torch.device = torch.device('cpu'), init_strategy: str = None) -> None:
        super().__init__(model_type = None, model_path = None, writer = writer, data = None, device = device, init_strategy = init_strategy)
        self.stack = embeddings
        if init_strategy is not None:
            print_if_debug("reseting network parameters...", self.cc.DEBUG)
            assert isinstance(init_strategy, str)
            if init_strategy == "full": self.stack = [torch.normal(mean = 0.0, std = 1., size = s.size()) for s in embeddings]
            else: print("No init with {} strategy".format(init_strategy))
        self.writer = writer
        self.data = None
        self.device = device

    def make_probe(self, prober: torch.nn.Module, enable_grads: bool = False, use_variational: bool = False, save_outputs: bool = False, task_title: dict = None) -> dict:

        def _prepare_data(batch):
            """Helper function
            """
            labels = batch['label'].to(self.device)
            inp_values = batch['input_values'].to(self.device)
            return inp_values, labels

        print_if_debug("stacking classifiers...", self.cc.DEBUG)

        loss_fn = Loss(use_variational)

        probing_info = {'loss': [], 'metrics': []}

        model_config = {'in_size': self.stack[0].size(-1), 
                        'hidden_size': 100,
                        'out_size': len(torch.unique(self.data['label'])),
                        'variational': use_variational,
                        'device': self.device}

        for ind, _ in enumerate(self.stack):
            self.logger.log_string(f"emb {ind} of {len(self.stack)} in process")     

            probing_model = prober(parent_model = torch.nn.Identity(self.stack[0].size(-1)),
                        clf = LinearModel(**model_config),
                        enable_grads = enable_grads,
                        ).to(self.device)
            probing_model.eval()
            tr = Trainer(model = probing_model.to(self.device), logger = self.logger, profiler = self.profiler, writer = self.writer,
                         loss_function = loss_fn, optimizer = torch.optim.Adam, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, device = self.device, lr = 3. * 1e-3)
            print_if_debug("training...", self.cc.DEBUG)
            _ = tr.train(train_loader = self.dataloader, batch_processing_fn = _prepare_data, count_of_epoch = self.cc.N_EPOCHS, info = {"layer": ind})
            

            if save_outputs:
                chkpnt = self.checkpointer(probing_model = probing_model.cpu(),
                                            task_title = "" if task_title is None else task_title['title'],
                                            params = model_config, layer_idx = ind, optimizer = tr.optimizer)
                print_if_debug("checkpoint {} saved...".format(chkpnt), self.cc.DEBUG)
            
            self._clear_cache()
            print_if_debug("validating...", self.cc.DEBUG)
            valid_loss, valid_metrics = tr.validate(valid_loader = self.validloader, batch_processing_fn = _prepare_data, metrics = F1Score(task_title["metrics"]), info = {"layer": ind})            
            probing_info['loss'].append(valid_loss)
            probing_info['metrics'].append(valid_metrics) 
            
            probing_model = probing_model.cpu()
            del probing_model
            self.logger.log_string(self.profiler.rep + "\n")

        print_if_debug('running probes...', self.cc.DEBUG)
        return probing_info