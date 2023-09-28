import numpy as np
import torch

def sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("./../../checkpoints/l2p_cifar100_repo.pt")
model = model.to(device)
keys = model.prompt.prompt_key.detach().cpu().numpy()
prompts = model.prompt.prompt.detach().cpu().numpy()
# prompts = prompts.reshape(-1, prompts.shape[-1])
# key_prompts = np.concatenate((keys, prompts), axis=0)

n_prompts = 10
length = 5

for key_id in range(n_prompts):
    print(f"Compute similarities for key {key_id}:")
    key = keys[key_id]
    for prompt_id in range(n_prompts):
        for token_id in range(length):
            prompt = prompts[prompt_id, token_id]
            s = sim(key, prompt)
            print(f"Key {key_id}, prompt {prompt_id}-{token_id}: {s}")
    print("\n")

for key_id_1 in range(n_prompts):
    for key_id_2 in range(n_prompts):
        if key_id_1 != key_id_2:
            key_1, key_2 = keys[key_id_1], keys[key_id_2]
            s = sim(key_1, key_2)
            print(f"Keys {key_id_1}-{key_id_2} similarity: {s}")
    print()


for prompt_id in range(n_prompts):
    print(f"Computing similarities of prompt {prompt_id}:")
    for token_1 in range(length):
        for token_2 in range(length):
            if token_1 != token_2:
                prompt_1 = prompts[prompt_id, token_1]
                prompt_2 = prompts[prompt_id, token_2]
                s = sim(prompt_1, prompt_2)
                print(f"Tokens {token_1}-{token_2} similarity: {s}")
    print("")