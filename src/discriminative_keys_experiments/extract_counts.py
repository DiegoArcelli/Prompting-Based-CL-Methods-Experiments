import sys
sys.path.append("./../")
import torch
from torchvision import transforms
from knn_l2p import KNNLearningToPrompt
# from avalanche.models.vit import create_model
import torchvision
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog='Compute key counts', description='Compute the key-class and key-task counts')
parser.add_argument('--model', default="avalanche", type=str, choices=["avalanche", "repo", "random", "repo_100"])
parser.add_argument('--dataset', default="train", type=str,  choices=["train", "test"])
args = parser.parse_args()


torch.cuda.set_per_process_memory_fraction(0.50)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seed = 42

scale = (0.05, 1.0)
ratio = (3. / 4., 4. / 3.)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

size = int((256 / 224) * 224)
eval_transform = transforms.Compose([
    transforms.Resize(size, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


batch_size = 16
num_classes = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

suffix = f"{args.dataset}"
if args.dataset == "test":
    dataset = torchvision.datasets.CIFAR100(root='./../../data/', train=False, download=True, transform=train_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
elif args.dataset == "train":
    dataset = torchvision.datasets.CIFAR100(root='./../../data/', train=True, download=True, transform=eval_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


if args.model == "repo":
    model_path = "./../../checkpoints/l2p_cifar100_repo.pt"
    suffix += "_repo"
elif args.model == "repo_100":
    model_path = "./../../checkpoints/l2p_cifar100_100_repo.pt"
    suffix += "_repo_100"
elif args.model == "avalanche":
    model_path = "./../../checkpoints/l2p_cifar100_l2p_selection.pt"
elif args.model == "random":
    model_path = "./../../checkpoints/l2p_cifar100_random.pt"
    suffix += "_random"

model_name="vit_base_patch16_224"

knn_l2p = KNNLearningToPrompt(
    model_path=model_path,
    model_name=model_name,
    pretrained=True,
    num_classes=100,
    num_tasks=10,
    predict_task=False,
    device=device,
)

with tqdm(total=len(data_loader)) as pbar:
    for batch_idx, (inputs, labels) in enumerate(data_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        knn_l2p.train(inputs, labels)
        pbar.update(1)

print(f"key_class_counts_{suffix} = {knn_l2p.key_class_counts}")
print(f"key_task_counts_{suffix} = {knn_l2p.key_task_counts}")