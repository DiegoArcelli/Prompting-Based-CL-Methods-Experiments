import torch
from torch.nn import CrossEntropyLoss
from prompt_selection_experiments.l2p import LearningToPrompt
from prompt_selection_experiments.vit_er import ViTER
from prompt_selection_experiments.vit_gdumb import ViTGDumb
from prompt_selection_experiments.vit_der import ViTDER
from prompt_selection_experiments.l2p_task_wise import TaskWiseLearningToPrompt
from prompt_selection_experiments.config import (
    l2p_no_selection_config,
    l2p_selection_config,
    l2p_offline_selection_config,
    l2p_offline_no_selection_config,
    l2p_task_wise_config,
    reply_selection_config,
    reply_no_selection_config,
    l2p_no_prompt_config
)
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def knn_l2p_forward(model, vit, x, key_class_mapping):

    with torch.no_grad():
        cls_features = vit(x)["pre_logits"]

    # normalize the key and the query
    keys = model.prompt.l2_normalize(model.prompt.prompt_key, dim=1) # (N_K, D)
    query = model.prompt.l2_normalize(cls_features, dim=1) # (B, D)
    similarity = torch.matmul(query, keys.T) # (B, N_K)

    # find the top k keys for each element of the batch
    _, idx = torch.topk(similarity, k=1, dim=1) # (B, K)

    # we map each one of the top k keys to its correspondent class   
    keys_preds = idx.detach().cpu().apply_(lambda x: key_class_mapping[x]).to(x.device) # (B, K)
    # classes = idx.detach().cpu().apply_(lambda x: key_class_map[x]).to(self.device) # (B, K)

    # compute the most present classes for each batch element
    pred = keys_preds.max(dim=1)[0] # (B)
    one_hot_pred = F.one_hot(pred, num_classes=100).float() # (B, C)

    return one_hot_pred



def get_strategy_arguments(parser, strategy_name="l2p", selection=False):

    assert strategy_name in ["l2p", "offline", "gdumb", "er", "der", "l2p_task_wise", "vit"]

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
    elif strategy_name == "l2p_task_wise":
        l2p_task_wise_config.get_args_parser(parser)
    elif strategy_name == "vit":
        l2p_no_prompt_config.get_args_parser(parser)

def get_strategy(strategy_name="l2p", strategy_args=None):

    assert strategy_name in ["l2p", "offline", "gdumb", "er", "der", "l2p_task_wise", "vit"]
    
    if strategy_name in ["l2p", "offline", "vit"]:
        cl_strategy = LearningToPrompt
    elif strategy_name == "gdumb":
        cl_strategy = ViTGDumb
    elif strategy_name == "er":
        cl_strategy = ViTER
    elif strategy_name == "der":
        cl_strategy = ViTDER
    elif strategy_name == "l2p_task_wise":
        cl_strategy = TaskWiseLearningToPrompt

    return cl_strategy(**strategy_args)


def get_knn_saliency_map(model, vit, image, label, key_class_mapping):

    criterion = CrossEntropyLoss()

    image.requires_grad_()
    logits = knn_l2p_forward(model, vit, image, key_class_mapping)
    loss = criterion(logits, label)
    # loss = F.nll_loss(logits, label, reduction='sum')
    model.zero_grad()
    vit.zero_grad()
    loss.backward()
    # grad = torch.autograd.grad(outputs = loss, inputs = image, only_inputs=True, retain_graph=False)[0]

    return torch.abs(image.grad).sum(1, keepdim=True)


def get_saliency_map(model, vit, image, label):

    criterion = CrossEntropyLoss()

    image.requires_grad_()
    logits = l2p_forward(model, vit, image)
    loss = criterion(logits, label)
    # loss = F.nll_loss(logits, label, reduction='sum')
    model.zero_grad()
    vit.zero_grad()
    grad = torch.autograd.grad(outputs = loss, inputs = image, only_inputs=True, retain_graph=False)[0]

    return torch.abs(grad).sum(1, keepdim=True)