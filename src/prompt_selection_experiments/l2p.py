from typing import Callable, List, Optional, Union
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import numpy as np
from timm.models.helpers import (
    checkpoint_seq,
    resolve_pretrained_cfg,
    build_model_with_cfg,
)
from avalanche.models.timm_vit import _load_weights, checkpoint_filter_fn
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, Block


class LearningToPrompt(SupervisedTemplate):

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
        prompt_selection: bool = True,
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
            prompt_selection=prompt_selection
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
            batch_size = logits.shape[0]
            for i in range(batch_size):
                label = self.mb_y[i].item() # get the true class of the i-th element of the batch
                # iterate over each prompt selected prompt for the i-th element of the batch
                for j in range(self.model.prompt.top_k): 
                    key_id = self.res["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                    # increase the key-class counter
                    task = label // 10
                    self.model.key_class_counts[key_id][label] += 1
                    self.model.key_task_counts[key_id][task] += 1

        if self.use_mask and self.is_training:
            mask = self.experience.classes_in_this_experience
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits


    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        if self.prompt_selection:
            loss = loss - self.sim_coefficient * self.res["reduce_sim"]
        return loss


class ViTWithPrompt(VisionTransformer):

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
        prompt_selection=True
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

        if pool_size != None:
            self.key_class_counts = {k: {c: 0 for c in range(num_classes)} for k in range(pool_size)}   
            self.key_task_counts = {k: {t: 0 for t in range(num_classes//10)} for k in range(pool_size)}

        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(
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
                prompt_selection=prompt_selection
            )

        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = x.float()
        x = self.patch_embed(x)

        if hasattr(self, "prompt"):
            if self.use_prompt_mask and train:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res["total_prompt_len"]
            x = res["prompted_embedding"]
        else:
            res = dict()
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        res["x"] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res["x"]
        if self.class_token and self.head_type == "token":
            x = x[:, 0]
        elif self.head_type == "gap" and self.global_pool == "avg":
            x = x.mean(dim=1)
        elif self.head_type == "prompt" and self.prompt_pool:
            x = (
                x[:, 1 : (1 + self.total_prompt_len)]
                if self.class_token
                else x[:, 0 : self.total_prompt_len]
            )
            x = x.mean(dim=1)
        elif self.head_type == "token+prompt" and self.prompt_pool and self.class_token:
            x = x[:, 0 : self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        res["pre_logits"] = x

        x = self.fc_norm(x)

        res["logits"] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(
            x, task_id=task_id, cls_features=cls_features, train=train
        )
        res = self.forward_head(res)
        return res

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)



class Prompt(nn.Module):

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
        prompt_selection=True
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
        self.prompt_selection = prompt_selection

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
            prompt_mean = torch.mean(self.prompt, dim=1).detach()
            self.prompt_key = prompt_mean


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm

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

            if self.prompt_selection:
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

            else:
                idx = torch.tensor([p for p in range(self.pool_size)])
                idx = idx.repeat(x_embed.shape[0], 1)
                reduce_sim = 0.0

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C

            out["prompt_idx"] = idx
            out["reduce_sim"] = reduce_sim
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


def vit_base_patch16_224(pretrained=True, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def create_model(model_name="", **kwargs):
    get_model = globals()[model_name]
    return get_model(**kwargs)


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for \
                            Vision Transformer models."
        )

    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        ViTWithPrompt,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs,
    )
    return model