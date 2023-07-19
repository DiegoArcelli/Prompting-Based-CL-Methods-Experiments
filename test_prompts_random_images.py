import torch
from utils import *

model = torch.load("./checkpoints/l2p_cifar_100_trained.pt")
model.eval()

rand_batch = torch.randn(8, 3, 224, 224)

for i in range(model.prompt.pool_size):
    res = prompt_forward(model, rand_batch, i)
    print(res["logits"].argmax(dim=1))
