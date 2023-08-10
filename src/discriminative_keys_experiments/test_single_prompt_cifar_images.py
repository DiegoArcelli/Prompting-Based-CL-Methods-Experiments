import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms
from math import ceil
from tqdm import tqdm
from avalanche.models.vit import create_model

def get_class_images(dataset, _class):
    filtered_dataset = list(filter(lambda x: x[1] == _class, dataset))
    return filtered_dataset

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 100

model = torch.load("./../../checkpoints/l2p_cifar100_repo.pt")
model = model.to(device)
model.eval()


vit = create_model(
    model_name="vit_base_patch16_224",
    pretrained=True,
    num_classes=n_classes,
    drop_rate=0.0,
    drop_path_rate=0.0,
).to(device)

vit.reset_classifier(0)

for p in vit.parameters():
    p.requires_grad = False

dataset = CIFAR100("./../data/", train=False, download=True)
# val_loader = DataLoader(dataset, batch_size=batch_size)

n_images = 128
batch_size = 16
pool_size = 10

for c in range(n_classes):
    pred_class = {x: {c: 0 for c in range(100)} for x in range(-1, pool_size+1)}
    class_dataset = get_class_images(dataset, c)
    class_dataset = list(map(lambda data: transform(data[0]), class_dataset))[:n_images]
    batched_dataset = torch.stack(class_dataset)
    iters = ceil(batched_dataset.size(0)/batch_size)

    with tqdm(total=(pool_size+2)*iters) as pbar:
        for prompt_id in range(-1, pool_size+1):
            for i in range(iters):
                x = batched_dataset[i*batch_size:(i+1)*batch_size].to(device)
                if prompt_id != pool_size:
                    out = prompt_forward(model, x, [prompt_id])
                    logits = out["logits"]
                else:
                    logits = l2p_forward(model, vit, x)
                preds = logits.argmax(dim=1)
                for pred in preds:
                    pred_class[prompt_id][pred.item()] += 1
                pbar.update(1)

    print(f"\nResults for class {c}:\n")
    for i in range(-1, pool_size+1):
        if i == -1:
            print("No prompt:")
        elif i == pool_size:
            print("L2P Prompts:")
        else:
            print(f"Prompt {i}")
        print(get_top_k_classes(pred_class[i], 10))
        print()
    print("\n")