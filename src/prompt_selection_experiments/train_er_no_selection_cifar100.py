import sys
sys.path.append("./../")
import torch
from torchvision import transforms
from vit_er import ViTER
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.nn import CrossEntropyLoss
from utils import count_parameters
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.benchmarks.generators import benchmark_with_validation_stream

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

text_logger = TextLogger(open("logs/log_er_no_selection.txt", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
)

early_stop = EarlyStoppingPlugin(
    patience=2,
    val_stream_name="valid_stream",
    verbose=True,
    mode="min",
    metric_name="Loss_Stream"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device="cpu"
num_classes=100

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

benchmark = benchmark_with_validation_stream(benchmark, 0.05, shuffle=True)

strategy = ViTER(
    model_name="vit_tiny_patch16_224",
    criterion=CrossEntropyLoss(),
    mem_size=5000,
    train_epochs=5,
    train_mb_size=16,
    eval_mb_size=16,
    device=device,
    prompt_pool=True,
    use_cls_features=False,
    prompt_selection=False,
    head_type="prompt",
    num_classes=num_classes,
    pool_size=10,
    prompt_length=5,
    top_k=10,
    sim_coefficient=0,
    evaluator=eval_plugin,
    plugins=[early_stop],
    eval_every=1
)

# print(strategy.model)
count_parameters(strategy.model)

train_stream = benchmark.train_stream
valid_stream = benchmark.valid_stream
test_stream = benchmark.test_stream

results = []
for t, (train_exp, valid_exp) in enumerate(zip(train_stream, valid_stream)):
    print("Start of experience: ", train_exp.current_experience)
    print("Current Classes: ", train_exp.classes_in_this_experience)
    strategy.train(train_exp, eval_streams=[valid_exp])
    results.append(strategy.eval(benchmark.test_stream[:t+1]))