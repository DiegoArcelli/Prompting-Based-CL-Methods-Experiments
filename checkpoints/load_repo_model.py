from avalanche.models.vit import create_model
import torch

# model_path = "./../../checkpoints/task10_checkpoint.pth"
model_path = "task10_checkpoint.pth"
checkpoint = torch.load(model_path)

num_classes=100

model_name="vit_base_patch16_224"
prompt_pool=True
pool_size=100
prompt_length=5
top_k=5
prompt_key=True
pretrained=True
num_classes=num_classes
drop_rate=0.0
drop_path_rate=0.0
embedding_key="cls"
prompt_init="uniform"
batchwise_prompt=False
head_type="prompt"
use_prompt_mask=False

model = create_model(
    model_name=model_name,
    prompt_pool=prompt_pool,
    pool_size=pool_size,
    prompt_length=prompt_length,
    top_k=top_k,
    prompt_key=prompt_key,
    pretrained=pretrained,
    num_classes=num_classes,
    drop_rate=drop_rate,
    drop_path_rate=drop_path_rate,
    embedding_key=embedding_key,
    prompt_init=prompt_init,
    batchwise_prompt=batchwise_prompt,
    head_type=head_type,
    use_prompt_mask=use_prompt_mask,
)
model.load_state_dict(checkpoint["model"])

torch.save(model, "l2p_100_cifar100_repo.pt")