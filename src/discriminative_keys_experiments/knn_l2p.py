from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import torch
from torch import nn
from avalanche.training.supervised import LearningToPrompt
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.problem_type import SupervisedProblem
from typing import Callable, Optional, List, Union
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.models.vit import create_model
import numpy as np
from functools import reduce
import torch.nn.functional as F
import numpy as np

class KNNLearningToPrompt(LearningToPrompt):

    def __init__(
        self,
        model : str = None,
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
        knn_mode: bool = False,
        predict_task: bool = True,
        **kwargs,
    ):
        
        if device is None:
            device = torch.device("cpu")

        self.num_classes = num_classes
        self.lr = lr
        self.sim_coefficient = sim_coefficient
        self.k = k

        if model is None:
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
            peval_mode
        )

        self._criterion = criterion
        self.use_cls_features = use_cls_features
        self.train_prompt_mask = train_prompt_mask
        self.use_mask = use_mask
        self.use_vit = use_vit
        self.knn_mode = knn_mode
        self.predict_task = predict_task

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

        if self.knn_mode:
            return self.knn_forward(logits, self.mb_y, cls_features)

        
        if self.use_mask and self.is_training:
            mask = self.experience.classes_in_this_experience
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits
    


    def knn_forward(self, logits, classes, cls_features):

        if self.knn_mode and self.is_training:

            # iterate over each prediction
            for i in range(classes.shape[0]):
                _class = classes[i].item() # get the predicted class
                # iterate over each prompt selected prompt for the i-th element of the batch
                for j in range(self.model.prompt.top_k): 
                    key_id = self.res["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                    # increase the key-class counter
                    task = _class // 10
                    self.model.key_class_counts[key_id][_class] += 1
                    self.model.key_task_counts[key_id][task] += 1
            return logits

        
        if self.knn_mode and self.is_eval:
            # normalize the selected keys
            keys = self.model.prompt.l2_normalize(self.model.prompt.prompt_key, dim=1) # (N_K, D)

            # normalize the query
            query = self.model.prompt.l2_normalize(cls_features, dim=1) # (B, D)

            # compute the cosine similarity between the queries and the keys
            similarity = torch.matmul(query, keys.T) # (B, N_K)

            # find the top k keys for each element of the batch
            _, idx = torch.topk(similarity, k=self.k, dim=1) # (B, K)

            # we map each one of the top k keys to its correspondent class    
            key_class_map = self.key_class_mapping()    

            if self.predict_task:
                self.mbatch[1] = self.mbatch[1] // self.num_tasks

            classes = idx.detach().cpu().apply_(lambda x: key_class_map[x]).to(self.device) # (B, K)

            # compute the most present classes for each batch element
            pred = classes.max(dim=1)[0] # (B)

            # compute the one hot representation of each class in the batch
            one_hot_pred = F.one_hot(pred, num_classes=self.num_classes) # (B, C)

            return one_hot_pred.float()
        

    def switch_to_knn_mode(self):
        if not self.knn_mode:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            self.knn_mode = True
            self.num_tasks = 10 if self.num_classes == 100 else 5

            self.model.key_class_counts = {k: {c: 0 for c in range(self.num_classes)} for k in range(self.model.prompt.pool_size)}
            self.model.key_task_counts = {k: {t: 0 for t in range(self.num_tasks)} for k in range(self.model.prompt.pool_size)}


    def key_class_mapping(self):
        # create a dict that maps a prompt key to its most predicted class
        key_class_map = {x: None for x in range(self.model.prompt.pool_size)}
        for i in range(self.model.prompt.pool_size):
            # select the class count dictionary for the i-th key of the pool
            if self.predict_task:
                sub_dict = self.model.key_task_counts[i]
            else:
                sub_dict = self.model.key_class_counts[i]
            # find in the class count dict the most predicted class for that key
            counts = list(sub_dict.values())    
            pred_class = np.argmax(counts)
            key_class_map[i] = pred_class
        return key_class_map
    
    
    def backward(self):
        if not self.knn_mode:
            self.loss.backward(retain_graph=self.retain_graph)