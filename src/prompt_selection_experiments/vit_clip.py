from typing import Callable, List, Optional, Union
from avalanche.training import LearningToPrompt
import torch
import torch.nn as nn
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ClippedLearningToPrompt(LearningToPrompt):


    def __init__(
        self,
        model_name: str,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        peval_mode: str = "epoch",
        prompt_pool: bool = True,
        pool_size: int = 20,
        prompt_length: int = 5,
        top_k: int = 5,
        lr: float = 0.03,
        sim_coefficient: float = 0.1,
        prompt_key: bool = True,
        pretrained: bool = True,
        num_classes: int = 10,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embedding_key: str = "cls",
        prompt_init: str = "uniform",
        batchwise_prompt: bool = False,
        head_type: str = "prompt",
        use_prompt_mask: bool = False,
        train_prompt_mask: bool = False,
        use_cls_features: bool = True,
        use_mask: bool = True,
        use_vit: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
            prompt_pool,
            pool_size,
            prompt_length,
            top_k,
            lr,
            sim_coefficient,
            prompt_key,
            pretrained,
            num_classes,
            drop_rate,
            drop_path_rate,
            embedding_key,
            prompt_init,
            batchwise_prompt,
            head_type,
            use_prompt_mask,
            train_prompt_mask,
            use_cls_features,
            use_mask,
            use_vit,
            **kwargs
        )

    def _after_backward(self, **kwargs):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)