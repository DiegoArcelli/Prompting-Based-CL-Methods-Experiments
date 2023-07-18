from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import torch
from torch import nn
from avalanche.training.supervised import GDumb, DER, LearningToPrompt
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from typing import Callable, Optional, Sequence, List, Union
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.models.vit import create_model
from avalanche.training.templates import SupervisedTemplate
from avalanche.models import Prompt
import numpy as np
from functools import reduce
import torch.nn.functional as F

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
    

class KNNLearning2Prompt(LearningToPrompt):

    def __init__(
        self,
        model_checkpoint : str = None,
        model_name : str = "vit_base_patch16_224",
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
        k : int = 1,
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
        
        if device is None:
            device = torch.device("cpu")

        self.num_classes = num_classes
        self.lr = lr
        self.sim_coefficient = sim_coefficient
        self.k = k

        if model_checkpoint is None:
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
                if n.startswith(tuple(["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])):
                    p.requires_grad = False

            model.head = torch.nn.Linear(model.head.in_features, num_classes).to(device)
            model.key_class_counts = {(k, c): 0 for k in range(model.prompt.pool_size) for c in range(num_classes)}
        else:
            model = torch.load(model_checkpoint)

        self.model = model
                
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

        SupervisedTemplate.__init__(
            self,
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self._criterion = criterion
        self.use_cls_features = use_cls_features
        self.train_prompt_mask = train_prompt_mask
        self.use_mask = use_mask
        self.use_vit = use_vit

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


    def forward(self):
        assert self.experience is not None
        if self.use_cls_features:
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
        else:
            self.res = {}
            self.res["logits"] = self.model(x=self.mb_x)
            self.res["reduce_sim"] = 0

        logits = self.res["logits"]

        if self.is_training:
            pred_classes = logits.argmax(dim=1)
            for i in range(pred_classes.shape[0]):
                pred_class = pred_classes[i].item()
                for j in range(self.model.prompt.top_k):
                    key_id = self.res["prompt_idx"][i][j].item()
                    self.model.key_class_counts[(key_id, pred_class)] += 1

        if self.is_eval and self.use_vit:
            keys = self.model.prompt.l2_normalize(self.model.prompt.prompt_key, dim=1)
            query = self.model.prompt.l2_normalize(cls_features, dim=1)
            similarity = torch.matmul(query, keys.T)
            _, idx = torch.topk(similarity, k=self.k, dim=1)
            key_class_map = self.key_class_mapping()
            idx.apply_(lambda x: key_class_map[x])
            pred = idx.max(dim=1)[0]
            one_hot_pred = F.one_hot(pred)
            return one_hot_pred

        
        if self.use_mask and self.is_training:
            mask = self.experience.classes_in_this_experience
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits
    

    def _after_training(self, **kwargs):
        torch.save(self.model, f"./checkpoints/l2p_cifar_100_trained.pt")
        return super()._after_training(**kwargs)
    
    def key_class_mapping(self):
        key_class_map = {x: None for x in range(self.model.prompt.top_k)}
        for i in range(self.model.prompt.top_k):
            sub_dict = {c: v for (k,c), v in self.model.key_class_counts.item() if k == i}
            pred_class, count = reduce(lambda a, x: x if x[1] > a[1] else a, sub_dict.items(), (0, 0))
            key_class_map[0] = pred_class
        return key_class_map
