import sys
sys.path.append("./../")
import torch
from utils import *
from avalanche.benchmarks.datasets import CIFAR100
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    transforms.PILToTensor(),
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_classes = 100

model = torch.load("./../../checkpoints/knn_l2p_tiny_cifar100.pt")
model.eval()

dataset = CIFAR100("./../data/", train=False, download=True)

batch_size = 8

for class_id in range(n_classes):
    x_pil = list(filter(lambda x: x[1] == class_id, dataset))[:batch_size]
    x_pil = list(map(lambda x: x[0], x_pil))
    x = list(map(lambda x: transform(x), x_pil))
    x = torch.stack(x)
    print(x.shape)
    break
    for prompt_id in range(model.prompt.pool_size):
        res = prompt_forward(model, x, prompt_id)
        print(res["logits"].argmax(dim=1))
    print("\n")