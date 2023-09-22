import torch
from avalanche.training import LearningToPrompt
from prompt_selection_experiments.vit_er import ViTER
from prompt_selection_experiments.vit_gdumb import ViTGDumb
from prompt_selection_experiments.vit_der import ViTDER
from torch.nn import CrossEntropyLoss
from prompt_selection_experiments.config import (
    l2p_no_selection_config,
    l2p_selection_config,
    l2p_offline_selection_config,
    l2p_offline_no_selection_config,
    reply_selection_config,
    reply_no_selection_config
)

def count_parameters(model):
    n_params_trainable =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'The model has {n_params} total parameters')
    print(f'The model has {n_params_trainable} trainable parameters')


def get_top_k_classes(preds, k : int):
    top_k = sorted(preds.items(), key=lambda x: x[1])[::-1][:k]
    non_zero_top_k =  [(_class, count) for (_class, count) in top_k if count != 0]
    formatted_top_k = [f"Class {_class}: {count}" for (_class, count) in non_zero_top_k]
    return ", ".join(formatted_top_k)

def adjust_pos_embed(x_len, pos_embed):
    pos_len = pos_embed.shape[1]
    dim = pos_embed.shape[-1]  
    new_pos_embed = torch.ones(1, x_len, dim).to(pos_embed.get_device())
    diff = abs(pos_len - x_len)

    if pos_len > x_len:
        new_pos_embed = pos_embed[:, diff:, :]
    elif pos_len < x_len:
        line = pos_embed[0][:1, :]
        lines = line.repeat(diff, 1).unsqueeze(0)
        new_pos_embed[:, :diff, :] = lines
        new_pos_embed[:, diff:, :] = pos_embed
    else:
        new_pos_embed = pos_embed

    return torch.nn.Parameter(new_pos_embed)

def prompt_forward_features(model, x, prompt_ids):

    x = x.float()
    x_embed = model.patch_embed(x)

    out = {}
    batch_size = x.shape[0]


    if prompt_ids != [-1]:
        prompts = model.prompt.prompt
        selected_prompts = prompts[prompt_ids]

        n_selected, length, dim = selected_prompts.shape

        selected_prompts = selected_prompts.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        selected_prompts = selected_prompts.reshape(batch_size, n_selected*length, dim)
        out["total_prompt_len"] = selected_prompts.shape[1]
        x = torch.cat([selected_prompts, x_embed], dim=1)
    else:
        out["total_prompt_len"] = 0
        x = x_embed

    model.total_prompt_len = out["total_prompt_len"]
    x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    model.pos_embed = adjust_pos_embed(x.shape[1], model.pos_embed)
    x = model.pos_drop(x + model.pos_embed)
    x = model.blocks(x)
    x = model.norm(x)
    out["x"] = x
    return out


def prompt_forward(model, x, prompt_ids):
    res = prompt_forward_features(model, x, prompt_ids)
    res = model.forward_head(res)
    return res


def l2p_forward(model, vit, x):

    with torch.no_grad():
        cls_features = vit(x)
    
    x_len = cls_features["x"].shape[1] + model.prompt.top_k*model.prompt.length
    model.pos_embed = adjust_pos_embed(x_len, model.pos_embed)
    res = model(
        x=x,
        task_id=-1,
        cls_features=cls_features["pre_logits"],
        train=None,
    )
    
    logits = res["logits"]

    return logits


def get_strategy_arguments(parser, strategy_name="l2p", selection=False):

    assert strategy_name in ["l2p", "offline", "gdumb", "er", "der"]

    if strategy_name=="l2p" and selection:
        l2p_selection_config.get_args_parser(parser)
    elif strategy_name=="l2p" and not selection:
        l2p_no_selection_config.get_args_parser(parser)
    elif strategy_name=="offline" and selection:
        l2p_offline_selection_config.get_args_parser(parser)
    elif strategy_name=="offline" and not selection:
        l2p_offline_no_selection_config.get_args_parser(parser)
    elif strategy_name in ["gdumb", "er", "der"] and selection:
        reply_selection_config.get_args_parser(parser)
    elif strategy_name in ["gdumb", "er", "der"] and not selection:
        reply_no_selection_config.get_args_parser(parser)


def get_strategy(strategy_name="l2p", strategy_args=None):

    assert strategy_name in ["l2p", "offline", "gdumb", "er", "der"]
    
    if strategy_name == "l2p" or strategy_name == "offline":
        cl_strategy = LearningToPrompt
    elif strategy_name == "gdumb":
        cl_strategy = ViTGDumb
    elif strategy_name == "er":
        cl_strategy = ViTER
    elif strategy_name == "der":
        cl_strategy = ViTDER

    return cl_strategy(**strategy_args)