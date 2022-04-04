from typing import Any
class TaskTester(object):
    """A class to make probing"""
    def __init__(self) -> list: 
        """grouping args according to argparser methodology"""
        self.auxillary_args = dict()
        self.model_args = dict()
        self.dataset_args = dict()
        self.experiment_args = dict()
    def _set_auxillary_args(self, auxillary_args: dict) -> None:
        self.auxillary_args = auxillary_args
    def _set_model_args(self, model_args: dict) -> None:
        self.model_args = model_args
    def _set_dataset_args(self, dataset_args: dict) -> None:
        self.dataset_args = dataset_args
    def _set_experiment_args(self, experiment_args: dict) -> None:
        self.experiment_args = experiment_args
    def __call__(self, *args: Any, **kwds: Any) -> dict:
        """You have to override this method to make this work"""
        raise NotImplementedError("")