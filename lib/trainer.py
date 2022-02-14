#TODO DOCUMENTATION!!!!!!

from typing import Callable, List, Any, Tuple, Union
import torch
from tqdm import tqdm
import numpy as np
from .profilers import MyLogger, ProbingProfiler
from .clf import Loss
import gc
from sklearn.metrics import f1_score
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
        return self.mfnc(torch.argmax(y_pred.cpu(), dim = -1).numpy(), y_true.cpu().numpy(), average = 'weighted')
    
class Trainer():
    """This class provides main train and validation functions"""
    def __init__(self, model: torch.nn.Module, logger: MyLogger, profiler: ProbingProfiler, writer: torch.utils.tensorboard.SummaryWriter, 
                       loss_function: Loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, 
                       device: torch.device, callback = None, lr: float = 1e-2) -> None:

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
      self.model =  model
      self.loss_function = loss_function
      self.optimizer = optimizer(self.model.parameters(), lr = lr)
      self.scheduler = scheduler(self.optimizer, T_max = 10)
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
        output = self.model(*x_batch if isinstance(x_batch, tuple) else x_batch)

        loss = self.loss_function(labels, output, self.model.clf)
        loss.backward()

        self.optimizer.step()
        prof.step()
        return loss.cpu().item()

    def _clear_cache(self):
        if self.device.type == 'cuda':
            with torch.no_grad(): torch.cuda.empty_cache()
            gc.collect()

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler) -> float:
        """Args:
            train_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
        Returns: loss: int, value of loss_fn on the batch of data
        """
        train_loss = []
        for it, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
            inputs, attention_masks, labels = batch_processing_fn(batch)
            batch_loss = self.train_on_batch((inputs, attention_masks), labels, prof)

            if self.callback is not None:
                with torch.no_grad(): self.callback(self.model, batch_loss)              
            train_loss.append(batch_loss)

        return np.mean(train_loss)

    @torch.no_grad()
    def valid_epoch(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, prof: ProbingProfiler, metrics: callable) -> Tuple:
        """Args:
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            prof, ProbingProfiler instance 
            metrics, callable: own callable metrics
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
            valid_loss.append(batch_loss)
            valid_metrics.append(metrics(output.detach().cpu(), labels.detach().cpu()))
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
        self.model.train()
        iterations = tqdm(range(count_of_epoch), desc = 'epoch')
        iterations.set_postfix({'train epoch loss': np.nan})
        self.logger.log_string(f"training...")
        with self.profiler.profile('train') as prof:
            for it in iterations:
                self.logger.log_string(f"{it} out of {count_of_epoch}")
                epoch_loss = self.train_epoch(train_loader = train_loader, batch_processing_fn = batch_processing_fn, prof = prof)
                
                self.writer.add_scalar("training loss of layer {}".format(info["layer"]), epoch_loss, it * len(train_loader))
                self.scheduler.step() 
                iterations.set_postfix({'train epoch loss': epoch_loss})
        self._clear_cache()
        return self


    @torch.no_grad()
    def validate(self, valid_loader: torch.utils.data.DataLoader, batch_processing_fn: Callable, metrics: Callable) -> Tuple:
        """Args:
            valid_loader, torch.utils.data.DataLoader
            batch_processing_fn, callable: a function for processing each batch
            metrics, callable: own callable metrics
        Returns: 
            valid_loss: float, value of loss_fn on the batch of data
            valid_metrics: float, value of metrics_fn on the batch of data
        """
        self.model.eval()
        self.logger.log_string(f"validating...")
        with self.profiler.profile('validation') as prof:
            valid_loss, valid_metrics = self.valid_epoch(valid_loader, batch_processing_fn, prof = prof, metrics = metrics)
        self._clear_cache()
        return valid_loss, valid_metrics
                        

