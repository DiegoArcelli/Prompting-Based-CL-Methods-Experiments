import numpy as np
import torch

def sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load("./../../checkpoints/knn_l2p_base_cifar100.pt")
model = model.to(device)
keys = model.prompt.prompt_key.detach().cpu().numpy()
prompts = model.prompt.prompt.detach().cpu().numpy()
# prompts = prompts.reshape(-1, prompts.shape[-1])
# key_prompts = np.concatenate((keys, prompts), axis=0)

n_prompts = 10
length = 5

for key_id in range(n_prompts):
    print(f"Compute similarity for key {key_id}:")
    key = keys[key_id]
    for prompt_id in range(n_prompts):
        for token_id in range(length):
            prompt = prompts[prompt_id, token_id]
            s = sim(key, prompt)
            print(f"Key {key_id}, prompt {prompt_id}-{token_id}: {s}")
    print("\n")