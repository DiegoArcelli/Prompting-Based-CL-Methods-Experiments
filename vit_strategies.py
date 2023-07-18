from avalanche.core import SupervisedPlugin
import torch
from avalanche.training.supervised import GDumb, DER
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from typing import Callable, Optional, Sequence, List, Union
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator

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

    def forward(self):
        return self.model(self.mb_x)["logits"]
    

class ViTDER(DER):

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch"
    ):
        
        super().__init__(
            model,
            optimizer,
            criterion,
            mem_size,
            batch_size_mem,
            alpha,
            beta,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode
        )

    def forward(self):
        return self.model(self.mb_x)["logits"]