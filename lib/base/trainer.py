#TODO DOCUMENTATION!!!!!!

from cProfile import label
from lib2to3.pytree import convert
from typing import Callable, List, Any, Tuple, Union
import torch
import torch.utils.tensorboard
import numpy as np
from .profilers import MyLogger, ProbingProfiler
from .layers import Loss
import gc
from sklearn.metrics import f1_score, accuracy_score

from .utils import test_ipkernel

if test_ipkernel(): from tqdm.notebook import tqdm 
else: from tqdm import tqdm

class CustomMetrics(object):
    """A class to add and invent your own callable metrics"""
    def __init__(self, metrics_name: str, fnct: Callable):
        """Args:
        """
        self.name  = metrics_name
        self.mfnc = fnct
    def __str__(self) -> str:
        return self.name
    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor, *args: Any, **kwds: Any) -> Union[List, float]:
        """Args:
                y_pred, torch.Tensor: outputs of neural networks (logits for regression, distribution for classification)
                y_true, torch.Tensor: groundtruth labels provided for the supervised learining tasks
            Returns:
                list (e.g. per classes) or float computed metrics
        """
        raise NotImplementedError("")

class F1Score(CustomMetrics):
    def __init__(self, metrics_name: str, fnct: Callable = f1_score):
        super().__init__(metrics_name, fnct)
    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor, *args: Any, **kwds: Any) -> Union[List, float]:
        assert y_pred.requires_grad == False and y_true.requires_grad == False

        if len(y_pred.shape) >= 2: return self.mfnc(torch.argmax(y_pred.cpu(), dim = -1).numpy(), y_true.cpu().numpy(), average = 'weighted')
        else: return self.mfnc(y_pred.cpu().numpy(), y_true.cpu().numpy(), average = 'weighted')
    
class FlattenF1Score(CustomMetrics):
    def __init__(self, metrics_name: str, fnct: Callable = f1_score):
        super().__init__(metrics_name, fnct)
    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor, *args: Any, **kwds: Any) -> Union[List, float]:
        assert y_pred.requires_grad == False and y_true.requires_grad == False
        return self.mfnc(y_pred.cpu().numpy(), y_true.cpu().numpy(), average = 'weighted')
    

class Trainer():
    """This class provides main train and validation functions"""
    def __init__(self, model: torch.nn.Module, logger: MyLogger, profiler: ProbingProfiler, writer: torch.utils.tensorboard.SummaryWriter, 
                       loss_function: Loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, 
                       device: torch.device, callback = None, lr: float = 1e-2, **kwargs) -> None:

      """
      Args:
        model, class inherited of nn.Module
        logger, MyLogger: own logger class instance
        profiler, ProbingProfiler: profiler
        writer, torch.utils.tensorboard.SummaryWriter: tensorboard default writer
        loss_fn, Loss: torch.nn.loss_fn for the task or inherited class
        optimizer, torch.optim.optimizer for the task
        scheduler, torch.optim.lr_scheduler: learning rate for the optimizer
        device, torch.device: a training device
        callback: an initialized object of callback class 
      """
      self.model =  model.to(device)
      self.loss_function = loss_function
      self.optimizer = optimizer(self.model.parameters(), lr = lr)
      self.scheduler = scheduler(self.optimizer, **kwargs) if scheduler else\
                       torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience = 2)
      self.callback = callback
      self.logger = logger
      self.device = device
      self.writer = writer
      self.profiler = profiler


    def train_on_batch(self, x_batch, labels, prof: ProbingProfiler) -> float:
        """
        This function needs to be implemented for 
        any particular model;
        Args:
            x_batch, labels: batches of data and target
            prof, ProbingProfiler instance 
        Output: loss: int, value of loss_fn on the batch of data
        """
        _ = self.model.train()
        
        self.optimizer.zero_grad()
        output = self.model(*x_batch if isinstance(x_batch, tuple) else (x_batch, None))
        loss = self.loss_function(labels.long(), output,  self.model.clf)
    
        loss.backward()
        self.optimizer.step()
        prof.step()
        return loss.cpu().item()

    def train_on_batch_ctc(self, x_batch, labels, prof: ProbingProfiler) -> float:
        """
        This function needs to be implemented for 
        any particular model;
        Args:
            x_batch, labels: batches of data and target
            prof, ProbingProfiler instance 
        Output: loss: int, value of loss_fn on the batch of data
        """
        _ = self.model.train()
        
        self.optimizer.zero_grad()
        output = self.model(*x_batch if isinstance(x_batch, tuple) else x_batch)
        input_lengths = torch.full(size=(labels.size(0),), fill_value=output.size(1), dtype=torch.long)
        target_lengths = torch.sum(labels >= 0, -1)
        loss = self.loss_function(labels.masked_select(labels >= 0), output, self.model.clf, 
                                  input_lengths = input_lengths, target_lengths = target_lengths)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 666)
        self.optimizer.step()
        prof.step()
        
        return loss.cpu().item()

    def _clear_cache(self):
        if self.device.type == 'cuda':
            with torch.no_grad(): torch.cuda.empty_cache()
            gc.collect()

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler, ctc: bool, attention_masks: bool =True) -> float:
        """Args:
            train_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
        Returns: loss: int, value of loss_fn on the batch of data
        """
        train_loss = []
        for it, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
            if attention_masks:
                inputs, attention_masks, labels = batch_processing_fn(batch)
                batch_loss = self.train_on_batch_ctc((inputs, attention_masks), labels, prof) if ctc else\
                            self.train_on_batch((inputs, attention_masks), labels, prof)       
            else:
                inputs, labels = batch_processing_fn(batch)
                batch_loss = self.train_on_batch(inputs, labels, prof)       
            if self.callback is not None:
                with torch.no_grad(): self.callback(self.model, batch_loss)              
            train_loss.append(batch_loss)
            self.scheduler.step(batch_loss) 


        return np.mean(train_loss)

    @torch.no_grad()
    def valid_epoch(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler, metrics: callable) -> Tuple:
        """Args:
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
            metrics, callable: own callable metrics
            output_convert_fn: a Callable output processor for non-auxillary tasks
        Returns: 
            valid_loss_per_loader: float, value of loss_fn on the batch of data
            valid_metrics_per_loader: float, value of metrics_fn on the batch of data
                """
        _ = self.model.eval()
        valid_loss, valid_metrics = [], []
        for it, batch in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            inputs, attention_masks, labels = batch_processing_fn(batch)
            output = self.model(inputs, attention_masks)
            batch_loss = self.loss_function(labels, output, self.model.clf)
            valid_loss.append(batch_loss.detach().cpu().item())
            valid_metrics.append(metrics(output.detach().cpu(), labels.detach().cpu()))
            prof.step()
            
        return np.mean(valid_loss), np.mean(valid_metrics)

    @torch.no_grad()
    def valid_epoch_ctc(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler, metrics: callable, **kwagrs) -> Tuple:
        """Args:
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
            metrics, callable: own callable metrics
            output_convert_fn: a Callable output processor for non-auxillary tasks
        Returns: 
            valid_loss_per_loader: float, value of loss_fn on the batch of data
            valid_metrics_per_loader: float, value of metrics_fn on the batch of data
                """
        _ = self.model.eval()
        valid_loss, valid_metrics = [], []
        for it, batch in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            inputs, attention_masks, labels = batch_processing_fn(batch)
            output = self.model(inputs, attention_masks)

            input_lengths = torch.full(size=(labels.size(0),), fill_value=output.size(1), dtype=torch.long)
            target_lengths = torch.sum(labels >= 0, -1)
            batch_loss = self.loss_function(labels.masked_select(labels >= 0), output, self.model.clf, 
                                  input_lengths = input_lengths, target_lengths = target_lengths)
            valid_loss.append(batch_loss.detach().cpu().item())
            

            cl_dim = output.size(-1)
            valid_metrics.append(metrics(output.detach().cpu().reshape(-1, cl_dim), torch.where(labels >= 0, labels, 0).detach().cpu().reshape(-1)))
            
            prof.step()
        return np.mean(valid_loss), np.mean(valid_metrics)
    

    def train(self, train_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, count_of_epoch: int, info: dict):
        """
        Trainer of the model;  uses `train_epoch` method
        Args:
            train_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            count_of_epoch, int: number of training epochs
            info, dict: some auxillary info
        Returns: self
        """
        self.model = self.model.to(self.device)
        self.model.train()
        iterations = tqdm(range(count_of_epoch), desc = 'epoch')
        iterations.set_postfix({'train epoch loss': np.nan})
        self.logger.log_string(f"training...")
        with self.profiler.profile('train') as prof:
            for it in iterations:
                self.logger.log_string(f"{it} out of {count_of_epoch}")
                epoch_loss = self.train_epoch(train_loader = train_loader, batch_processing_fn = batch_processing_fn, prof = prof, 
                                              ctc = info['ctc'], attention_masks = info['attention_masks'])
                
                self.writer.add_scalar("training loss of layer {}".format(info["layer"]), epoch_loss, it * len(train_loader))
                iterations.set_postfix({'train epoch loss': epoch_loss})
                self._clear_cache()
        return self


    @torch.no_grad()
    def validate(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, metrics: Callable, info: dict, **kwagrs) -> Tuple:
        """Args:
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            metrics, callable: own callable metrics
            info, dict: some auxillary info
            output_convert_fn: a Callable output processor for non-auxillary tasks
        Returns: 
            valid_loss: float, value of loss_fn on the batch of data
            valid_metrics: float, value of metrics_fn on the batch of data
        """
        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.log_string(f"validating...")
        with self.profiler.profile('validation') as prof:
            if info["attention_masks"]:
                valid_loss, valid_metrics = self.valid_epoch_ctc(valid_loader, batch_processing_fn, prof = prof, metrics = metrics, **kwagrs) if info['ctc'] else\
                                            self.valid_epoch(valid_loader, batch_processing_fn, prof = prof, metrics = metrics)
            else: valid_loss, valid_metrics = self._valid_epoch(valid_loader, batch_processing_fn, prof = prof, metrics = metrics)
            self.writer.add_scalar("valid loss", valid_loss, info['layer'])
            self.writer.add_scalar(f"valid {metrics.name}", valid_metrics, info['layer'])

        self._clear_cache()
        return valid_loss, valid_metrics
                
    @torch.no_grad()
    def _valid_epoch(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler, metrics: callable) -> Tuple:
        """Args: (simple function for embedding probing)
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
            metrics, callable: own callable metrics
            output_convert_fn: a Callable output processor for non-auxillary tasks
        Returns: 
            valid_loss_per_loader: float, value of loss_fn on the batch of data
            valid_metrics_per_loader: float, value of metrics_fn on the batch of data
                """
        _ = self.model.eval()
        valid_loss, valid_metrics = [], []
        for it, batch in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            inputs, labels = batch_processing_fn(batch)
            output = self.model(inputs, None)
            batch_loss = self.loss_function(labels, output, self.model.clf)
            valid_loss.append(batch_loss.detach().cpu().item())
            valid_metrics.append(metrics(output.detach().cpu(), labels.detach().cpu()))
            prof.step()
            
        return np.mean(valid_loss), np.mean(valid_metrics)
