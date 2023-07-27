import sys
sys.path.append("./../")
import torch
from torchvision import transforms
from vit_gdumb import ViTGDumb
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.nn import CrossEntropyLoss
from utils import count_parameters
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training import GDumb
from avalanche.models import SimpleCNN


text_logger = TextLogger(open("log.txt", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
)



early_stop = EarlyStoppingPlugin(
    patience=1,
    val_stream_name="valid_stream",
    verbose=True,
)

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
        eval_transform=eval_transform,
        shuffle=True
    )
else:
    benchmark = SplitCIFAR100(
        n_experiences=10,
        seed=42,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
        shuffle=True
    )

benchmark = benchmark_with_validation_stream(benchmark, 0.05, shuffle=True)

strategy = ViTGDumb(
    model_name="vit_tiny_patch16_224",
    criterion=CrossEntropyLoss(),
    mem_size=100,
    train_epochs=100,
    train_mb_size=2,
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
    head_type="prompt",
    use_prompt_mask=False,
    train_prompt_mask=False,
    use_cls_features=True,
    use_mask=False,
    use_vit=True,
    lr = 0.03,
    sim_coefficient = 0.1,
    plugins=[early_stop],
    eval_every=1
)

count_parameters(strategy.model)

results = []
for train_experience, valid_experience in zip(benchmark.train_stream, benchmark.valid_stream):
    print("Start of experience: ", train_experience.current_experience)
    print("Current Classes: ", train_experience.classes_in_this_experience)
    strategy.train(train_experience, eval_streams=[valid_experience])
    # strategy.eval()
    # strategy.eval(benchmark.valid_stream[t])
results.append(strategy.eval(benchmark.test_stream))
