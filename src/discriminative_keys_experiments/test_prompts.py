import sys
sys.path.append("./../")
import torch
# from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from avalanche.models.vit import create_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def adjust_pos_embed(x, pos_embed):
    pos_len = pos_embed.shape[1]
    x_len = x.shape[1]
    dim = pos_embed.shape[-1]  
    new_pos_embed = torch.ones(1, x_len, dim).to(device)
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

    prompts = model.prompt.prompt
    selected_prompts = prompts[prompt_ids]

    n_selected, length, dim = selected_prompts.shape

    selected_prompts = selected_prompts.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    selected_prompts = selected_prompts.reshape(batch_size, n_selected*length, dim)
    out["total_prompt_len"] = selected_prompts.shape[1]
    model.total_prompt_len = out["total_prompt_len"]
    x = torch.cat([selected_prompts, x_embed], dim=1)
    x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    model.pos_embed = adjust_pos_embed(x, model.pos_embed)
    x = model.pos_drop(x + model.pos_embed)
    x = model.blocks(x)
    x = model.norm(x)
    out["x"] = x
    return out

def prompt_forward(model, x, prompt_ids):
    res = prompt_forward_features(model, x, prompt_ids)
    res = model.forward_head(res)
    return res



model = torch.load("./../../checkpoints/knn_l2p_base_cifar100.pt")
model = model.to(device)
model.eval()

x = torch.randn(4, 3, 224, 224).to(device)
res = prompt_forward(model, x, [0, 2])
print(res.keys(), res["x"].shape)