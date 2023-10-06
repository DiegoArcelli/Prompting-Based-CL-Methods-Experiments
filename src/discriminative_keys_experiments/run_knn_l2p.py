import sys
sys.path.append("./../")
import torch
from torchvision import transforms

from knn_l2p import KNNLearningToPrompt
# from avalanche.models.vit import create_model
import torchvision
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog='KNN L2P', description='KNN L2P')
parser.add_argument('--model', default="avalanche", type=str, choices=["avalanche", "repo", "random", "repo_100"])
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

train_set = torchvision.datasets.CIFAR100(root='./../../data/', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR100(root='./../../data/', train=False, download=True, transform=eval_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

if args.model == "avalanche":
    predict_task = True
    max_k = 10
    model_path = "./../../checkpoints/l2p_cifar100_l2p_selection.pt"
elif args.model == "repo":
    max_k = 10
    predict_task = True
    model_path = "./../../checkpoints/l2p_cifar100_repo.pt"
elif args.model == "repo_100":
    max_k = 100
    predict_task = False
    model_path = "./../../checkpoints/l2p_cifar100_100_repo.pt"
elif args.model == "random":
    max_k = 10
    predict_task = True
    model_path = "./../../checkpoints/l2p_cifar100_random.pt"

model_name="vit_base_patch16_224"

knn_l2p = KNNLearningToPrompt(
    model_path=model_path,
    model_name=model_name,
    pretrained=True,
    num_classes=100,
    num_tasks=10,
    predict_task=predict_task,
    device=device,
)

with tqdm(total=len(train_loader)) as pbar:
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        knn_l2p.train(inputs, labels)
        pbar.update(1)


print(knn_l2p.key_class_counts)
print(knn_l2p.key_task_counts)

knn_l2p.compute_key_class_mapping()

for k in range(1, max_k+1):
    total_acc = 0.0
    knn_l2p.k = k
    with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if knn_l2p.predict_task:
                labels = labels // knn_l2p.num_tasks

            out = knn_l2p.predict(inputs)
            acc = sum(out == labels)
            total_acc += acc
            pbar.update(1)
    print(f"Final accuracy for k={k}: {total_acc/len(test_set)}")