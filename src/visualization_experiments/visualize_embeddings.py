import sys
sys.path.append("./../discriminative_keys_experiments/")
sys.path.append("./../prompt_selection_experiments/")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='Visualize L2P', description='Visualize L2P keys and prompts')
parser.add_argument('--checkpoint_name', default="l2p_cifar100_selection.pt", type=str)
parser.add_argument('--reduction_alg', default="tsne", type=str)

args = parser.parse_args()

output_dir = "../../plots/visualization"

def get_embs(vectors, alg="tsne"):
    assert alg in ["tsne", "pca"], "The selected dimensionality reduction algorithms doesn't exist"

    embs = None
    if alg == "tsne":
        embs = TSNE(n_components=2, n_iter=10000, learning_rate='auto', init='random', perplexity=3).fit_transform(vectors)
    elif alg == "pca":
        embs = PCA(n_components=2).fit_transform(vectors)
    return embs

alg = args.reduction_alg
# colors = list(mcolors.TABLEAU_COLORS.values())
colors = ["red", "blue", "yellow", "green", "sienna", "indigo", "darkorange", "magenta", "cyan", "lightgrey"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_path = f"./../../checkpoints/{args.checkpoint_name}"
model = torch.load(model_path)

# model = torch.load("./../../checkpoints/knn_l2p_cifar100_test_lr_batchwise.pt")
model = model.to(device)

keys = model.prompt.prompt_key.detach().cpu().numpy()
prompts = model.prompt.prompt.detach().cpu().numpy()
prompts = prompts.reshape(-1, prompts.shape[-1])
keys_prompts = np.concatenate((keys, prompts), axis=0)

alg = "tsne"

# visualize only keys
embs = get_embs(keys, alg)
fig, ax = plt.subplots()
for i in range(10):
    plt.scatter(embs[i, 0], embs[i, 1], marker="*", s=100, c=colors[i], label=f"Key {i}", edgecolors="black")
plt.legend()
# plt.show()
plt.savefig(f"{output_dir}/keys_embedding.png")
plt.clf()

# visualize only embeddings
embs = get_embs(prompts, alg)
for i in range(50):
    plt.scatter(embs[i, 0], embs[i, 1], s=20, label=(f"Prompts {i//5}" if i % 5 == 0 else None), c=colors[i//5]) #label=f"Key {i}", edgecolors="black")
plt.legend()
# plt.show()
plt.savefig(f"{output_dir}/prompt_embedding.png")
plt.clf()

# visualize keys+embeddings
embs = get_embs(keys_prompts, alg)
for i in range(10):
    plt.scatter(embs[i, 0], embs[i, 1], marker="*", s=100, c=colors[i], label=f"Key {i}", edgecolors="black")
    plt.scatter(embs[10 + i*5: 10 + (i+1)*5, 0], embs[10 + i*5: 10 + (i+1)*5, 1], c=colors[i], s=20)
plt.legend()
# plt.show()
plt.savefig(f"{output_dir}/keys_prompts_embedding.png")
plt.clf()
