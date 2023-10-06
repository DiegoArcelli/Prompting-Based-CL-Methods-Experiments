import sys
sys.path.append("./../")
sys.path.append('./../prompt_selection_experiments/')
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm
from prompt_selection_experiments.l2p import create_model
import numpy as np
import itertools


def get_class_images(dataset, _class):
    filtered_dataset = list(filter(lambda x: x[1] == _class, dataset))
    return filtered_dataset


size = int((256 / 224) * 224)
transform = transforms.Compose([
    transforms.Resize(size, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 100

# model_path = "./../../checkpoints/l2p_cifar100_l2p_selection.pt"
model_path = "./../../checkpoints/l2p_cifar100_repo.pt"
model = torch.load(model_path)
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
iters = images_per_class // batch_size + int(images_per_class % batch_size != 0)

prompts = list(itertools.combinations([x for x in range(10)], 5))
n_comb = len(prompts)
prompts = ["no prompt", "l2p", "random"] + [p for p in prompts]



total_iters = n_classes*iters*(n_comb+3)


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
                    model.head_type = "token"
                    logits = prompt_forward(model, batch, [])["logits"]
                    model.head_type = "prompt"
                elif prompt == "l2p":
                    logits = l2p_forward(model, vit, batch)
                elif prompt == "random":
                    random_prompts = list(np.random.choice(range(10), 5, replace=False))
                    logits = prompt_forward(model, batch, random_prompts)["logits"]
                else:
                    logits = prompt_forward(model, batch, list(prompt))["logits"]
                # print(logits)
                preds = logits.argmax(dim=1)
                for pred in preds:
                    accuracies[prompt][c] += int(pred.item() == c)
                pbar.update(1)
            


best_score, best = 0, None
worst_score, worst = 0, None
acc_list = []
for prompt in prompts:
    
    if prompt in ["no prompt", "l2p", "random"]:
        continue

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
    acc_list.append(tot_acc)

avg_score = sum(acc_list)/len(acc_list)

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


print(f"Average prompts accuracy: {avg_score}")
print(f"Best prompt {best} accuracy: {best_score}")
print(f"Worst prompt {worst} accuracy: {worst_score}")
print(f"No prompt accuracy: {no_prompt_score}")
print(f"L2P prompt accuracy: {l2p_score}")
print(f"Random prompt accuracy: {random_score}")
