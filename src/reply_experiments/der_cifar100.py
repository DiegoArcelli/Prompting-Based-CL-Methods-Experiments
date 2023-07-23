from torch.nn import Sequential, Linear, ReLU, Softmax
from torchvision import transforms
from avalanche.benchmarks import SplitCIFAR100
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models.vit import create_model
from reply_experiments.vit_strategies import ViTDER


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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


benchmark = SplitCIFAR100(
    n_experiences=10,
    seed=42,
    fixed_class_order=[c for c in range(100)],
    return_task_id=False,
    train_transform=train_transform,
    eval_transform=eval_transform
)

model = create_model(
    model_name="vit_tiny_patch16_224",
    img_size=224,
    in_chans=3,
    num_classes=100,
    # prompt__pool=True,
    # pool_size=10,
    # prompt_length=5,
    # top_k=5
)

for name, param in model.named_parameters():
    if name.startswith(tuple(["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])):
        param.requires_grad = False

emb_size = model.head.in_features
n_classes = model.head.out_features

model.head = Sequential(
    Linear(emb_size, 200),
    ReLU(),
    Linear(200, n_classes),
    Softmax()
)

strategy = ViTDER(
    model=model,
    optimizer=SGD(model.parameters(), lr=0.1, momentum=0.99),
    criterion=CrossEntropyLoss(),
    mem_size=5000,
    train_epochs=2,
    train_mb_size=8,
    eval_mb_size=2,
    device=device,
)

results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    strategy.train(experience)
    results.append(strategy.eval(benchmark.test_stream))

