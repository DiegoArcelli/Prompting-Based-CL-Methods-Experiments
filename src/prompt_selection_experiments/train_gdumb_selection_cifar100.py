import sys
sys.path.append("./../")
import torch
from torchvision import transforms
from vit_gdumb import ViTGDumb
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models.vit import create_model
from utils import count_parameters

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device="cpu"
num_classes=10

if num_classes == 10:
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=42,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
else:
    benchmark = SplitCIFAR100(
        n_experiences=10,
        seed=42,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform
    )

strategy = ViTGDumb(
    model_name="vit_tiny_patch16_224",
    criterion=CrossEntropyLoss(),
    mem_size=100,
    train_epochs=1,
    train_mb_size=8,
    eval_mb_size=2,
    device=device,
    num_classes=num_classes,
    prompt_selection=True,
    prompt_pool=True,
    pool_size=10,
    prompt_length=5,
    top_k=5,
    prompt_key=True,
    pretrained=True,
    embedding_key="cls",
    prompt_init="uniform",
    batchwise_prompt=False,
    head_type="token+prompt",
    use_prompt_mask=False,
    train_prompt_mask=False,
    use_cls_features=True,
    use_mask=False,
    use_vit=True,
    lr = 0.03,
    sim_coefficient = 0.5
)

count_parameters(strategy.model)

results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    strategy.train(experience)
results.append(strategy.eval(benchmark.test_stream))
