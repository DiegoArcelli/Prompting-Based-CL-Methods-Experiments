import sys
sys.path.append("./../")
import torch
from utils import *
from tqdm import tqdm

torch.cuda.set_per_process_memory_fraction(0.50)
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

pool_size = model.prompt.pool_size
n_classes = 100

# rand_batch = torch.randn(16, 3, 224, 224)

pred_class = {x: {c: 0 for c in range(100)} for x in range(-1, pool_size)}

n_iters = 5

prompt_ids = [-1]+list(range(0,10))

with tqdm(total=n_iters*(pool_size+1)) as pbar:
    for _ in range(n_iters):
        rand_batch = torch.rand(2, 3, 224, 224).to(device)
        for i in prompt_ids:
            res = prompt_forward(model, rand_batch, [i])
            preds = res["logits"].argmax(dim=1)
            for pred in preds:
                pred_class[i][pred.item()] += 1
            pbar.update(1)
        

for i in prompt_ids:
    if i != -1:
        print(f"Prompt {i}")
    else:
        print("No prompt:")
    print(print_top_k_classes(pred_class[i], 10))