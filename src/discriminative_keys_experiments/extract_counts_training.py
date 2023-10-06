import sys
sys.path.append("./../")
sys.path.append('./../prompt_selection_experiments/')
from prompt_selection_experiments import l2p
import torch

model_path = "./../../checkpoints/l2p_cifar100_l2p_selection.pt"
model = torch.load(model_path)
print("Avalanche model:")
print(model.key_class_counts)
print(model.key_task_counts)

print("\n\n\nRepository model:")
model_path = "./../../checkpoints/l2p_cifar100_repo.pt"
model = torch.load(model_path)
print(model.key_class_counts)
print(model.key_task_counts)