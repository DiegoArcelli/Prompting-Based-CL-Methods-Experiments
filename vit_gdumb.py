import torch
from avalanche.training.supervised import GDumb
from torch.nn import Module
from torch.optim import Optimizer
from typing import Callable, Optional, Sequence, List, Union
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator

class ViTGDumb(GDumb):

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        
        super().__init__(
            model,
            optimizer,
            criterion,
            mem_size,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

    # def criterion(self):
    #     return self._criterion(self.mb_output["logits"], self.mb_y)
    
    def forward(self):
        return self.model(self.mb_x)["logits"]