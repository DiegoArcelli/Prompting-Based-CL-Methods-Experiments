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
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block
from avalanche.models.timm_vit import ViTWithPrompt
from avalanche.models import Prompt

class RandomLearningToPrompt(LearningToPrompt):

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

        if device is None:
            device = torch.device("cpu")

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

        model.head = torch.nn.Linear(768, num_classes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

        super().__init__(
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

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

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

        print(self.res["prompt_idx"])
        logits = self.res["logits"]

        if self.use_mask and self.is_training:
            mask = self.experience.classes_in_this_experience
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        return loss


class ViTWithRandomPrompt(ViTWithPrompt):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        prompt_length=None,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        head_type="token",
        use_prompt_mask=False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )

        self.class_token = class_token
        num_patches = self.patch_embed.num_patches

        if no_embed_class:
            embed_len = num_patches
        else:
            embed_len = num_patches + self.num_prefix_tokens

        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask

        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = RandomPrompt(
                length=prompt_length,
                embed_dim=embed_dim,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                random_prompt=True
            )

        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)



class RandomPrompt(Prompt):

    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        random_prompt=True
    ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.random_prompt = random_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == "zero":
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == "uniform":
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        """
        Args:
            x_embed: input tensor
            prompt_mask: mask to select specific prompts.
            cls_features: key features to find the close prompts
        """
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == "mean":
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == "max":
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == "mean_max":
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(
                    x_embed, dim=1
                )
            elif self.embedding_key == "cls":
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError(
                    "Not supported way of calculating embedding keys!"
                )

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(
                        idx, return_counts=True, sorted=True
                    )
                    # In jnp.unique, when the 'size' is specified and there are
                    # fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value',
                    # the default is the minimum value along the specified
                    # dimension.
                    # Unless dimension is specified, this will be flattend if it
                    # is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat(
                            [
                                prompt_id,
                                torch.full(
                                    (self.pool_size - prompt_id.shape[0],),
                                    torch.min(idx.flatten()),
                                    device=prompt_id.device,
                                ),
                            ]
                        )
                        id_counts = torch.cat(
                            [
                                id_counts,
                                torch.full(
                                    (self.pool_size - id_counts.shape[0],),
                                    0,
                                    device=id_counts.device,
                                ),
                            ]
                        )
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C

            out["prompt_idx"] = idx

            # Debugging, return sim as well
            out["prompt_norm"] = prompt_norm
            out["x_embed_norm"] = x_embed_norm
            out["similarity"] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out["selected_key"] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out["reduce_sim"] = reduce_sim
        elif self.random_prompt:
            if self.batchwise_prompt:
                prompt_ids = np.random.choice(range(self.pool_size), self.top_k, replace=False)
                prompt_ids = np.expand_dims(prompt_ids, 0)
                prompt_ids = np.repeat(prompt_ids, batch_size, axis=0)
            else:
                prompt_ids = np.zeros((batch_size, top_k))
                prompt_ids = np.apply_along_axis(lambda _: np.random.choice(range(self.pool_size), self.top_k, replace=False), 1, prompt_ids)
            prompt_ids = torch.from_numpy(prompt_ids)
            batched_prompt_raw = self.prompt[prompt_ids]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C
            out["prompt_idx"] = prompt_ids
        else:
            if self.prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token,
        # C]
        out["total_prompt_len"] = batched_prompt.shape[1]
        out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
