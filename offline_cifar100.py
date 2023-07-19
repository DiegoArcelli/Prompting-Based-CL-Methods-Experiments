import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from avalanche.models.vit import create_model
import numpy as np
from tqdm import tqdm


# Hyperparameters
batch_size = 8
learning_rate = 0.03
num_epochs = 10
validation_ratio = 0.1


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"

# CIFAR-100 dataset
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))
np.random.seed(42)
np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

# DataLoaders for training and validation sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)



test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=eval_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = create_model(
    model_name="vit_tiny_patch16_224",
    img_size=224,
    in_chans=3,
    num_classes=100,
    pretrained=True
    # prompt__pool=True,
    # pool_size=10,
    # prompt_length=5,
    # top_k=5
).to(device)

for name, param in model.named_parameters():
    if name.startswith(tuple(["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])):
        param.requires_grad = False



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):

    train_loss = 0

    model.train()
    print(f"Training epoch {epoch + 1}")
    with tqdm(total=len(train_loader)) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            res = model(images)
            outputs = res["logits"]
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                train_loss /= 100
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {train_loss:.4f}')
                train_loss = 0

            pbar.update(1)


    model.eval()
    print(f"Validation epoch {epoch + 1}")
    with torch.no_grad():
        correct = 0
        total = 0
        with tqdm(total=len(val_loader)) as pbar:
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                res = model(images)
                outputs = res["logits"]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)

        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')


print("Testing")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    with tqdm(total=len(test_loader)) as pbar:
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            res = model(images)
            outputs = res["logits"]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.update(1)

    print(f'Accuracy on the test images: {100 * correct / total:.2f}%')