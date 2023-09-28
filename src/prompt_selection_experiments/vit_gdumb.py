from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import torch
from torch import nn
from avalanche.training.supervised import GDumb
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin, GDumbPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.models.vit import create_model
import numpy as np
from functools import reduce

from typing import (
    Callable,
    Optional,
    List,
    Union
)


class ViTGDumb(GDumb):

    def __init__(
        self,
        model_name: str,
        criterion=CrossEntropyLoss(),
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
        prompt_pool: bool = True,
        prompt_selection: bool = False,
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
        use_mask: bool = False,
        use_vit: bool = True,
        **base_kwargs
    ):

        self.num_classes = num_classes
        self.lr = lr
        self.sim_coefficient = sim_coefficient
        model = create_model(
            model_name=model_name,
            prompt_pool=prompt_pool,
            pool_size=pool_size,
            prompt_length=prompt_length,
            top_k=top_k,
            prompt_key=prompt_key,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            batchwise_prompt=batchwise_prompt,
            head_type=head_type,
            use_prompt_mask=use_prompt_mask,
        )

        for n, p in model.named_parameters():
            if n.startswith(
                tuple(["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])
            ):
                p.requires_grad = False

        model.head = torch.nn.Linear(model.head.in_features, num_classes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )
        
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

        self._criterion = criterion
        self.use_cls_features = use_cls_features
        self.train_prompt_mask = train_prompt_mask
        self.use_mask = use_mask
        self.use_vit = use_vit
        self.prompt_selection = prompt_selection

        if use_cls_features:
            self.original_vit = create_model(
                model_name=model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            ).to(device)

            self.original_vit.reset_classifier(0)

            for p in self.original_vit.parameters():
                p.requires_grad = False


    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

    def forward(self):
        assert self.experience is not None

        if self.use_cls_features and not self.prompt_selection:
            with torch.no_grad():
                cls_features = self.original_vit(self.mb_x)["pre_logits"]
        else:
            cls_features = None

        if self.use_vit:
            self.res = self.model(
                x=self.mb_x,
                task_id=self.mb_task_id,
                cls_features=cls_features,
                train=self.train_prompt_mask,
            )
            
            if not hasattr(self.model, "prompt"):
                self.res["reduce_sim"] = 0

        else:
            self.res = {}
            self.res["logits"] = self.model(x=self.mb_x)
            self.res["reduce_sim"] = 0

        logits = self.res["logits"]

        if self.use_mask and self.is_training:
            gdumb_plugin = list(filter(lambda plugin: type(plugin) == GDumbPlugin, self.plugins))[0]
            # mask = self.experience.classes_in_this_experience
            mask = list(gdumb_plugin.storage_policy.seen_classes)
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits
    

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        loss = loss - self.sim_coefficient * self.res["reduce_sim"]
        return loss

    def _after_backward(self, **kwargs):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)