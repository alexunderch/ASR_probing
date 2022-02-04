import psutil
import GPUtil as GPU
import torch
class DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): pass
    def __enter__(self, *args, **kwargs): return self
    def __exit__(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): pass

class ProbingProfiler:
    def __init__(self, log_dir: str): 
        assert isinstance(log_dir, str) and log_dir.endswith("/")
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