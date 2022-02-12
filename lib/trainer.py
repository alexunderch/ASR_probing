import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


class CustomMetrics(object):
    pass

class callback():
    def __init__(self, writer, dataset, loss_function, delimeter = 100, batch_size = 64, model_name = "LSTM", custom_tok_type = "LaBSE"):
        """
        This class provides a callback for Traniner class:
        Parameters:
        -- writer: TensorBoard writer to display stats to:
        -- dataset: torch.utils.data.Dataset for the task
        -- loss_fn: torch.nn.loss_fn for the task or inherited class
        -- delimetet: int, if (#step of optimizer % delim == 0) -> plot results
        -- batch_size: int, size of batch of data
        -- model_name: str, name of file to save

        """
        self.step = 0
        self.writer = writer
        self.delimeter = delimeter
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.model_name = model_name
        self.custom_tok_type = custom_tok_type
        self.dataset = dataset
    def save_model(self, model, name: str):
        """
        save model method; Parameters: 
        -- model: class inherited of nn.Module
        -- name: str, name of file to save
        Output: torch checkpoint
        """
        checkpoint = {'state_dict': model.state_dict(),
                      'loss_dict': self.loss_function.state_dict()}
        with open(name, 'wb') as f: torch.save(checkpoint, f)

    def forward(self, model, loss):

        """
        The main method of class: provides a working function;
        Parameters:
        -- model: class inherited of nn.Module
        -- loss_fn: torch.nn.loss_fn for the task or inherited class
        """
        # raise NotImplementedError("IMPLEMENT IT!")
        self.step += 1
        self.writer.add_scalar('LOSS/train', loss, self.step)
        
        if self.step % self.delimeter == 0:
            # _ = self.save_model(model, str(self.model_name) + "_iter_" + str(self.step))

            batch_generator = DataLoader(dataset = self.dataset, batch_size = self.batch_size)
            
            test_loss = 0
            _ = model.eval()
                      
            pred = []
            real = []
            for it, (x_batch, y_batch) in enumerate(batch_generator):
                x_batch = x_batch.to(model.device)
                y_batch = y_batch.to(model.device)

                with torch.no_grad():
                  output = model(x_batch)

                  test_loss += self.loss_function(output.reshape(-1, output.shape[-1]), y_batch.reshape(-1)).cpu().item() * len(x_batch)

                  pred.extend(torch.argmax(output, dim = -1).cpu().numpy().tolist())
                  real.extend(y_batch.cpu().numpy().tolist())

            test_loss /= len(self.dataset)
            
            self.writer.add_scalar('LOSS/test', test_loss, self.step)


    def __call__(self, model, loss):
        return self.forward(model, loss)

class Trainer():
    def __init__(self, model,  loss_function, optimizer, callback = None, lr = 1e-2):

      """
      This class provides main train and validation functions
      Parameters:
      -- model: class inherited of nn.Module
      -- loss_fn: torch.nn.loss_fn for the task or inherited class
      -- optimizer: torch.optim.optimizer for the task
      -- callback: an initialized object of callback class 
      """
      self.model =  model
      self.loss_function = loss_function
      self.optimizer = optimizer(self.model.parameters(), lr = lr)
      self.callback = callback

    def train_on_batch(self, x_batch, y_batch):
      """
      This function is need to be implemented for 
      any particular model;
      Parameters:
      -- x_batch, y_batch: batches of data and target
      Output: loss: int, value of loss_fn on the batch of data
      """
      # raise NotImplementedError("IMPLEMENT IT!")
      _ = self.model.train()
      
      self.optimizer.zero_grad()
      output = self.model(x_batch.to(self.model.device))

      loss = self.loss_function(output.reshape(-1, output.shape[-1]), y_batch.to(self.model.device).reshape(-1))
      loss.backward()

      self.optimizer.step()
      return loss.cpu().item()

    def train_epoch(self, train_generator):
      """
      method of train for batches of data on 1 epoch; 
      uses train_on_batch method
      Parameters:
      -- train_generator: torch on own batch generator
      Output:
      mean loss per the epoch: int
      """
      epoch_loss = 0
      total = 0
      for it, (batch_x, batch_y) in enumerate(train_generator):
          batch_loss = self.train_on_batch(batch_x, batch_y)
  
  
          if self.callback is not None:
              with torch.no_grad():
                  self.callback(self.model, batch_loss)
              
          epoch_loss += batch_loss * len(batch_x)
          total += len(batch_x)
      
      return epoch_loss/total

    def train(self, dataset, count_of_epoch, batch_size):
      """
      Trainer of the model; 
      uses train_epoch method
      Parameters:
      -- dataset: torch.utils.data.Dataset for the task or inherited class
      -- count_of_epoch: int, how many epochs to train
      -- batch_size: int, size of batch of data
      Output: self
      """
      _ = self.model.train()
      iterations = tqdm(range(count_of_epoch), desc = 'epoch')
      iterations.set_postfix({'train epoch loss': np.nan})
      for it in iterations:
          batch_generator = tqdm(
              DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True), 
              leave = False, total = len(dataset) // batch_size + (len(dataset) % batch_size > 0))
          
          epoch_loss = self.train_epoch(train_generator = batch_generator)
          
          iterations.set_postfix({'train epoch loss': epoch_loss})
      return self
