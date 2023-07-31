import sys
sys.path.append("./../")
import torch
from utils import *
from tqdm import tqdm
import itertools

torch.cuda.set_per_process_memory_fraction(0.50)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load("./../../checkpoints/knn_l2p_tiny_cifar100.pt")
model = model.to(device)
model.eval()

pool_size = model.prompt.pool_size
n_classes = 100

# rand_batch = torch.randn(16, 3, 224, 224)
prompts = list(itertools.combinations([x for x in range(10)], 5))
comb = len(prompts)
pred_class = {x: {c: 0 for c in range(100)} for x in range(comb)}

n_iters = 2

with tqdm(total=n_iters*comb) as pbar:
    for _ in range(n_iters):
        rand_batch = torch.rand(8, 3, 224, 224).to(device)
        for i, prompt in enumerate(prompts):
            res = prompt_forward(model, rand_batch, list(prompt))
            preds = res["logits"].argmax(dim=1)
            for pred in preds:
                pred_class[i][pred.item()] += 1
            pbar.update(1)
        

for i, prompt in enumerate(prompts):
    print(f"Prompt {prompt}")
    print(print_top_k_classes(pred_class[i], 10))
    print("\n")