import torch
from torchvision import transforms
from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10
from torch.nn import CrossEntropyLoss
from knn_l2p import KNNLearningToPrompt
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics


torch.cuda.set_per_process_memory_fraction(0.50)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seed = 42


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

text_logger = TextLogger(open("logs/log_task_knn_l2p.txt", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
)

num_classes = 100

if num_classes == 10:
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=seed,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
        shuffle=True
    )
else:
    benchmark = SplitCIFAR100(
        n_experiences=10,
        seed=seed,
        fixed_class_order=[c for c in range(num_classes)],
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
        shuffle=True
    )


model = torch.load("./../../checkpoints/l2p_cifar100_repo.pt")

strategy = KNNLearningToPrompt(
    model=model,
    model_name="vit_base_patch16_224",
    criterion=CrossEntropyLoss(),
    train_mb_size=16,
    eval_mb_size=16,
    device=device,
    train_epochs=1,
    num_classes=num_classes,
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
    use_cls_features=True,
    use_mask=True,
    use_vit=True,
    lr = 0.03,
    sim_coefficient = 0.1,
    drop_rate=0.0,
    drop_path_rate=0.0,
    k=1,
    seed=seed,
    evaluator=eval_plugin,
    predict_task=True
    # eval_every=1
)

strategy.switch_to_knn_mode()

# compute key class mapping and use keys and knn classifier
results = []
for experience in benchmark.train_stream:
    strategy.train(experience)

for k in range(1, 11):
    print(f"Results for k={k}")
    strategy.k = k
    results.append(strategy.eval(benchmark.test_stream))

print(strategy.model.key_class_counts)
print(strategy.model.key_task_counts)
