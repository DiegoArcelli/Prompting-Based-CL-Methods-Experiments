import sys
sys.path.append("./../")
import torch
from utils import *
from tqdm import tqdm

# torch.cuda.set_per_process_memory_fraction(0.50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("./../../checkpoints/l2p_cifar100_repo.pt")
model = model.to(device)
model.eval()

pool_size = model.prompt.pool_size
n_classes = 100

# rand_batch = torch.randn(16, 3, 224, 224)

pred_class = {x: {c: 0 for c in range(100)} for x in range(-1, pool_size)}

n_iters = 100
prompt_ids = [-1]+list(range(0,10))

with tqdm(total=n_iters*(pool_size+1)) as pbar:
    for _ in range(n_iters):
        rand_batch = torch.rand(16, 3, 224, 224).to(device)
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
    print(get_top_k_classes(pred_class[i], 10))