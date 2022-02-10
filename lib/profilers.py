import psutil
import GPUtil as GPU
import torch
import logging
import os
import json
import numpy as np
from datetime import datetime

from .constants import Constants

class DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): pass
    def __enter__(self, *args, **kwargs): return self
    def __exit__(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): pass

class ProbingProfiler:
    def __init__(self, log_dir: str): 
        assert isinstance(log_dir, str) 
        self.log_dir = log_dir
        self.prof = None
        self.rep = ""
        self._on = True
    def monitor_resources(self):
        self.rep = "virual memory usage: {} \n\n".format(str(psutil.virtual_memory())) +\
              "RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB \n".format(psutil.virtual_memory().free, 
                                                                                                   psutil.virtual_memory().used, 
                                                                                                   psutil.virtual_memory().percent, 
                                                                                                   psutil.virtual_memory().total) 
        gpu_info = lambda gpu: "GPU RAM Free: {0:.0f}MB \t|\t Used: {1:.0f}MB \t|\t Util {2:3.0f}% \t|\t Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal)
        self.rep += "\n".join([gpu_info(gpu) for gpu in  GPU.getGPUs()])
        print(self.rep)
        
    def off(self): self._on = False
    def on(self): self._on = True
        
    def __repr__(self): return self.rep
    
    def _config(self, wait = 1, warmup = 1, active = 4, repeat = 3):
        self.thr = (wait + warmup + active) * repeat
        self.schedule = torch.profiler.schedule(wait = wait, warmup = warmup, active = active, repeat = repeat)
    def profile(self, sub_dir: str = None, traceboard: bool = True):
        self._config()

        dir = self.log_dir
        if sub_dir is not None: dir += sub_dir 
    
        self.prof = torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                           schedule = self.schedule,
                                           on_trace_ready = torch.profiler.tensorboard_trace_handler(dir) if traceboard else None,
                                           record_shapes = True,
                                           profile_memory = True,
                                           with_stack = True)
        return self.prof if self._on else DummyClass()

class MyLogger(object):
    def __init__(self, filepath: str):
        self.fpath = filepath
        cc =Constants
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO if not cc.DEBUG else logging.DEBUG)
        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.DEBUG if cc.DEBUG else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    def log_string(self, message: str):
        assert isinstance(message, str)
        self.logger.info(message) 

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CheckPointer:
    def __init__(self, parent_dir: str): 
        """
        Class to save necessary model checkpoints and load them
        -> saving: full model, layer index, optimizer state dict 
        """
        cc = Constants
        assert isinstance(parent_dir, str)
        self.parent_dir = os.path.join(parent_dir, "checkpoints", str(cc.TODAY))

    def __call__(self, probing_model: torch.nn.Module, task_title: str,
                       params: dict, layer_idx: int, optimizer: torch.optim):
        """Classical torch checkpointing callable method
        Args:
            probing_model, torch.nn.Module: probing model to save
            params, dict: model configuration for inference

            layer_idx, int: index of hidden layer of the model
            optimizer, torch.optim: optional optimizer to save
        Returns:
            str, name of saved checkpoint
        checkpoint name format: CHKPNT_lyridx=int_DATETIME.pth
        """
        torch.save({
            'layer_index': layer_idx,
            'model_params': params,
            'model_state_dict': probing_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
            os.path.join(self.parent_dir, "probing_checkpoint_{}_lyridx={}.pth".format(task_title, layer_idx)))
        return self.parent_dir + "checkpoints/{}/probing_checkpoint_{}_lyridx={}.pth".format(datetime.now().strftime('%Y-%m-%d'),
                                                                                             task_title,
                                                                                             layer_idx)