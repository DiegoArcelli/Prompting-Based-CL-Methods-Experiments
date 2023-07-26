import torch
from torchvision import transforms
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.nn import CrossEntropyLoss
from knn_l2p import KNNLearningToPrompt
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
import os

torch.cuda.set_per_process_memory_fraction(0.5)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seed = 42

text_logger = TextLogger(open("logs/knn_l2p.txt", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
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
# device = "cpu"

num_classes = 10

if num_classes == 10:
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=seed,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
else:
    benchmark = SplitCIFAR100(
        n_experiences=10,
        seed=seed,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform
    )


strategy = KNNLearningToPrompt(
            model=None,
            model_name='vit_tiny_patch16_224',
            criterion=CrossEntropyLoss(),
            train_mb_size=24,
            eval_mb_size=24,
            device=device,
            train_epochs=5,
            num_classes=num_classes,
            prompt_pool=True,
            pool_size=10,
            prompt_length=5,
            top_k=5,
            prompt_key=True,
            pretrained=True,
            embedding_key="cls",
            prompt_init="uniform",
            batchwise_prompt=True,
            head_type="prompt",
            use_prompt_mask=False,
            # train_mask=True,
            use_cls_features=True,
            use_mask=True,
            use_vit=True,
            lr = 0.03,
            sim_coefficient = 0.1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            k=3,
            seed=seed
        )


results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    strategy.train(experience)
    results.append(strategy.eval(benchmark.test_stream))

torch.save(strategy.model, "./../../checkpoints/knn_l2p_cifar100.pt")