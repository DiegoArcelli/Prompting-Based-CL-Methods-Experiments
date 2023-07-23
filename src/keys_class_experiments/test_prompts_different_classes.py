import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor()
])

model = torch.load("./../../checkpoints/l2p_cifar_100_trained.pt")
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