import sys
sys.path.append("./../")
sys.path.append('./../prompt_selection_experiments/')
sys.path.append('./../discriminative_keys_experiments/')

import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms
from prompt_selection_experiments.l2p import create_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from keys_usage import key_class_counts_train_repo_100
from functools import reduce
import argparse

parser = argparse.ArgumentParser(prog='Visualize L2P', description='Visualize L2P keys and prompts')
parser.add_argument('--model', default="repo_100", type=str, choices=["avalanche", "repo", "random"])
args = parser.parse_args()


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
n_images = 2

# model_path = "./../../../checkpoints/l2p_cifar100_l2p_selection.pt"

if args.model == "repo_100":
    model_path = "./../../checkpoints/l2p_cifar100_100_repo.pt"
    key_class_counts = key_class_counts_train_repo_100


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
vit.eval()

vit.reset_classifier(0)

for p in vit.parameters():
    p.requires_grad = False

key_class_mapping = {key: reduce(lambda agg, x: agg if agg[1] > x[1] else x, list(counts.items()), (0,0))[0] for key, counts in key_class_counts.items()}

dataset = CIFAR100("./../../data/", train=True, download=True)

with tqdm(total=n_classes*n_images) as pbar:
    for c in range(n_classes):

        images = get_class_images(dataset, c)[:n_images]
        images = list(map(lambda data: transform(data[0]), images))
        images = torch.stack(images)

        for i in range(n_images):
            image = images[i].unsqueeze(0)
            label = torch.tensor([c])

            image = image.to(device)
            label = label.to(device)

            grad_map = get_knn_saliency_map(model, vit, image, label, key_class_mapping)

            grad_map = grad_map[0].permute(1, 2, 0).detach().cpu().numpy()
            img = image[0].permute(1, 2, 0).detach().cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img)
            ax1.axis('off')

            ax2.imshow(grad_map)
            ax2.axis('off')

            plt.subplots_adjust(wspace=0.05)

            # plt.savefig(f"./../../plots/{args.model}/saliency_maps/class_{c}/image_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            pbar.update(1)

        
