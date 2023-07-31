import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms
from math import ceil
from tqdm import tqdm
import itertools

def get_class_images(dataset, _class):
    filtered_dataset = list(filter(lambda x: x[1] == _class, dataset))
    return filtered_dataset

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: x.float()),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_classes = 100

model = torch.load("./../../checkpoints/knn_l2p_tiny_cifar100.pt")
model = model.to(device)
model.eval()

dataset = CIFAR100("./../data/", train=False, download=True)
# val_loader = DataLoader(dataset, batch_size=batch_size)

n_images = 20
batch_size = 4
ref_class = 0
pool_size = 10

prompts = list(itertools.combinations([x for x in range(pool_size)], 5))
comb = len(prompts)

pred_class = {x: {c: 0 for c in range(n_classes)} for x in range(comb)}
class_dataset = get_class_images(dataset, ref_class)
class_dataset = list(map(lambda data: transform(data[0]), class_dataset))[:n_images]
batched_dataset = torch.stack(class_dataset)
iters = ceil(batched_dataset.size(0)/batch_size)
with tqdm(total=comb*iters) as pbar:
    for idx, prompt_ids in enumerate(prompts):
        for i in range(iters):
            x = batched_dataset[i*batch_size:(i+1)*batch_size].to(device)
            out = prompt_forward(model, x, [prompt_ids])
            preds = out["logits"].argmax(dim=1)
            for pred in preds:
                pred_class[idx][pred.item()] += 1
            pbar.update(1)

print(f"Results for class {ref_class}:")
for i, prompt in enumerate(prompts):
    print(f"Prompt {prompt}")
    print(get_top_k_classes(pred_class[i], 10))
    print("\n")