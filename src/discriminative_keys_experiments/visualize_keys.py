from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load("./../../checkpoints/knn_l2p_base_cifar100.pt")
model = model.to(device)
keys = model.prompt.prompt_key.detach().cpu().numpy()
prompts = model.prompt.prompt.detach().cpu().numpy()
prompts = prompts.reshape(-1, prompts.shape[-1])
key_prompts = np.concatenate((keys, prompts), axis=0)


# embs = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(key_prompts)
embs = PCA(n_components=2).fit_transform(key_prompts)
fig, ax = plt.subplots()

colors = list(mcolors.TABLEAU_COLORS.values())

for i in range(10):
    plt.scatter(embs[i, 0], embs[i, 1], marker="*", s=100, c=colors[i], label=f"Key {i}", edgecolors="black")
    plt.scatter(embs[10 + i*5: 10 + (i+1)*5, 0], embs[10 + i*5: 10 + (i+1)*5, 1], c=colors[i])
plt.legend()
plt.show()