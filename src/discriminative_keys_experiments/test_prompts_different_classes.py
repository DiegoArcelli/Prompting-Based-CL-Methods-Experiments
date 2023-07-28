import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms

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

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = torch.load("./../../checkpoints/l2p_cifar_base_100_trained.pt")
model.eval()

dataset = CIFAR100("./../data/", train=False, download=True)

batch_size = 8

for class_id in range(10):
    x_pil = list(filter(lambda x: x[1] == class_id, dataset))[:batch_size]
    x_pil = list(map(lambda x: x[0], x_pil))
    x = list(map(lambda x: transform(x), x_pil))
    x = torch.stack(x)
    for prompt_id in range(model.prompt.pool_size):
        res = prompt_forward(model, x, prompt_id)
        print(res["logits"].argmax(dim=1))
    print("\n")