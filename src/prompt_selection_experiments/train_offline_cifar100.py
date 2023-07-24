import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from avalanche.models.vit import create_model
import numpy as np
from timm.models.vision_transformer import vit_tiny_patch16_224
from tqdm import tqdm


# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
validation_ratio = 0.1


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"

# CIFAR-100 dataset
train_transform = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
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

train_dataset = torchvision.datasets.CIFAR100(root='./../../data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(root='./../../data', train=False, download=True, transform=eval_transform)


val_size = int(len(train_dataset)*validation_ratio)
train_size = len(train_dataset) - val_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

emb_size = model.head.in_features
n_classes = model.head.out_features

model.head = nn.Linear(emb_size, n_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):

    total, total_loss, total_corr = 0, 0, 0

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
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            
            total_corr += (predicted == labels).sum().item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                cur_acc = total_corr / total
                cur_loss = total_loss / total
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {cur_loss:.4f}, Accuracy: {cur_acc:.4f}')

            pbar.update(1)

    print(f'Accuracy on the train images: {100 * total_corr / total:.2f}%')


    model.eval()
    print(f"Validation epoch {epoch + 1}")
    with torch.no_grad():
        total, total_loss, total_corr = 0, 0, 0
        with tqdm(total=len(val_loader)) as pbar:
            for i, (images, labels) in enumerate(val_loader):

                images, labels = images.to(device), labels.to(device)
                res = model(images)
                loss = criterion(outputs, labels)

                outputs = res["logits"]
                _, predicted = torch.max(outputs.data, 1)

                total_loss += loss.item()
                total += labels.size(0)
                total_corr += (predicted == labels).sum().item()
                pbar.update(1)

                if (i + 1) % 10 == 0:
                    cur_acc = total_corr / total
                    cur_loss = total_loss / total
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {cur_loss:.4f}, Accuracy: {cur_acc:.4f}')

        val_accuracy = 100 * total_corr / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')


print("Testing")
model.eval()
with torch.no_grad():

    total, total_loss, total_corr = 0, 0, 0
    with tqdm(total=len(test_loader)) as pbar:
        for i, (images, labels) in enumerate(test_loader):

            images, labels = images.to(device), labels.to(device)

            res = model(images)
            outputs = res["logits"]
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_corr += (predicted == labels).sum().item()
            total_loss = loss.item()
            pbar.update(1)

            if (i + 1) % 10 == 0:
                cur_acc = total_corr / total
                cur_loss = total_loss / total
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {cur_loss:.4f}, Accuracy: {cur_acc:.4f}')


    print(f'Accuracy on the test images: {100 * total_corr / total:.2f}%')