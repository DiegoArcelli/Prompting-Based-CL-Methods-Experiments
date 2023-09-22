import sys
sys.path.append("./../")
import torch
from torchvision import transforms
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.nn import CrossEntropyLoss
from avalanche.training import LearningToPrompt
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.benchmarks.generators import benchmark_with_validation_stream
import argparse
from utils import get_strategy, get_strategy_arguments

parser = argparse.ArgumentParser(prog='Train L2P', description='Train L2P based continual learning strategy')
parser.add_argument('-sel', '--selection', action='store_true')
parser.add_argument('-name', '--strategy_name', default="l2p", type=str)

args, _ = parser.parse_known_args()
args = get_strategy_arguments(parser, args.strategy_name, args.selection)
args = parser.parse_args()

non_strategy_args = ["selection, strategy_name", "num_experiences"]
strategy_args = {arg: value for (arg, value) in args._get_kwargs() if arg not in non_strategy_args}

use_early_stop = False
num_classes = 100
lr = 0.03
batch_size = 16

torch.cuda.set_per_process_memory_fraction(0.5)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seed = 42

log_name = f"log_l2p_{args.strategy_name}_{'selection' if args.selection else 'no_selection'}.txt"
text_logger = TextLogger(open(f"logs/{log_name}", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
)

strategy_args["evaluator"] = eval_plugin
strategy_args["train_mb_size"] = batch_size
strategy_args["eval_mb_size"] = batch_size
del strategy_args["batch_size"]


if use_early_stop:
    early_stop = EarlyStoppingPlugin(
        patience=2,
        val_stream_name="valid_stream",
        verbose=True,
        mode="min",
        metric_name="Loss_Stream"
    )
    strategy_args["plugins"] = [early_stop]
    strategy_args["eval_every"] = 1
else:
    strategy_args["plugins"] = None
    strategy_args["eval_every"] = -1

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strategy_args["device"] = device

strategy_args["lr"] = lr*batch_size/256.0


benchmark = SplitCIFAR100(
    n_experiences=args.num_experiences,
    seed=seed,
    fixed_class_order=[c for c in range(num_classes)],
    return_task_id=False,
    train_transform=train_transform,
    eval_transform=eval_transform,
    shuffle=True
)


if use_early_stop:
    benchmark = benchmark_with_validation_stream(benchmark, 0.05, shuffle=True)

print("Arguments:")
for arg, val in strategy_args.items():
    print(f"{arg} = {val}")
print("\n")


strategy = get_strategy(strategy_name=args.strategy_name, strategy_args=strategy_args)

train_stream = benchmark.train_stream
valid_stream = benchmark.valid_stream if use_early_stop else [None for _ in train_stream]
test_stream = benchmark.test_stream

results = []
for t, (train_exp, valid_exp) in enumerate(zip(train_stream, valid_stream)):
    print("Start of experience: ", train_exp.current_experience)
    print("Current Classes: ", train_exp.classes_in_this_experience)
    if use_early_stop:
        strategy.train(train_exp, eval_streams=[valid_exp])
    else:
        strategy.train(train_exp)
    results.append(strategy.eval(benchmark.test_stream[:t+1]))


ckpt_name = f"l2p_cifar100_{args.strategy_name}_{'selection' if args.selection else 'no_selection'}.pt"
torch.save(strategy.model, f"./../../checkpoints/{ckpt_name}")