import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms
from math import ceil
from tqdm import tqdm
from avalanche.models.vit import create_model
import numpy as np


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

dataset = CIFAR100("./../../data/", train=False, download=True)
# val_loader = DataLoader(dataset, batch_size=batch_size)


batch_size = 16
images_per_class = 100
pool_size = 10
iters = images_per_class // batch_size + int(images_per_class % batch_size != 0)

prompts = ["no prompt", "l2p", "random"] + [x for x in range(pool_size)]


# prompts = list(itertools.combinations([x for x in range(pool_size)], 5))
# comb = len(prompts)

total_iters = n_classes*iters*(pool_size+3)


# predictions = {prompt: {label: {_class: 0 for _class in range(n_classes)} for label in range(n_classes)} for prompt in prompts}
accuracies = {prompt: {label: 0 for label in range(n_classes)} for prompt in prompts}


with tqdm(total=total_iters) as pbar:
    for c in range(n_classes):
        class_images = get_class_images(dataset, c)
        class_dataset = list(map(lambda data: transform(data[0]), class_images))
        class_dataset = torch.stack(class_dataset)
        for i in range(iters):
            batch = class_dataset[i*batch_size:(i+1)*batch_size].to(device)
            for prompt in prompts:
                if prompt == "no prompt":
                    logits = prompt_forward(model, batch, [])["logits"]
                elif prompt == "l2p":
                    logits = l2p_forward(model, vit, batch)
                elif prompt == "random":
                    random_prompts = list(np.random.choice(range(10), 5, replace=False))
                    logits = prompt_forward(model, batch, random_prompts)["logits"]
                else:
                    logits = prompt_forward(model, batch, [prompt for _ in range(5)])["logits"]
                # print(logits)
                preds = logits.argmax(dim=1)
                for pred in preds:
                    accuracies[prompt][c] += int(pred.item() == c)
                pbar.update(1)
            


best_score, best = 0, None
worst_score, worst = 0, None
for prompt in range(pool_size):
    print(f"Accuracies for prompt {prompt}:")
    tot_corr = 0
    for c in range(n_classes):
        class_acc = accuracies[prompt][c]
        tot_corr += class_acc
        print(f"Class {c} accuracy: {class_acc/images_per_class}")
    tot_acc = tot_corr / (images_per_class*n_classes)

    if best is not None:
        if tot_acc > best_score:
            best = prompt
            best_score = tot_acc
    else:
        best = prompt
        best_score = tot_acc


    if worst is not None:
        if tot_acc < worst_score:
            worst = prompt
            worst_score = tot_acc
    else:
        worst = prompt
        worst_score = tot_acc
        
    print(f"Total accuracy: {tot_acc}\n\n")

no_prompt_score, random_score, l2p_score = (0, 0, 0)

print(f"Accuracy for no prompt:")
tot_corr = 0
for c in range(n_classes):
    class_acc = accuracies["no prompt"][c]
    tot_corr += class_acc
    print(f"Class {c} accuracy: {class_acc/images_per_class}")
tot_acc = tot_corr / (images_per_class*n_classes)
no_prompt_score = tot_acc
print(f"Total accuracy: {tot_acc}\n\n")


print(f"Accuracy for random prompts:")
tot_corr = 0
for c in range(n_classes):
    class_acc = accuracies["random"][c]
    tot_corr += class_acc
    print(f"Class {c} accuracy: {class_acc/images_per_class}")
tot_acc = tot_corr / (images_per_class*n_classes)
random_score = tot_acc
print(f"Total accuracy: {tot_acc}\n\n")


print(f"Accuracy for l2p prompts:")
tot_corr = 0
for c in range(n_classes):
    class_acc = accuracies["l2p"][c]
    tot_corr += class_acc
    print(f"Class {c} accuracy: {class_acc/images_per_class}")
tot_acc = tot_corr / (images_per_class*n_classes)
l2p_score = tot_acc
print(f"Total accuracy: {tot_acc}\n\n")


print(f"Best prompt {best} accuracy: {best_score}")
print(f"Worst prompt {worst} accuracy: {worst_score}")