import matplotlib.pyplot as plt
from tqdm import tqdm
from keys_usage import *
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='Visualize L2P', description='Visualize L2P keys and prompts')
parser.add_argument('--model', default="avalanche", type=str, choices=["avalanche", "repo", "random", "repo_100"])
parser.add_argument('--dataset', default="train", type=str,  choices=["train", "test", "training"])

args = parser.parse_args()

num_keys = 10 if args.model != "repo_100" else 100
num_classes = 100
num_tasks = 10
plots_dir = "./../../plots"


if args.model == "repo":
    dir = "repo/"
    if args.dataset == "train":
        key_class_counts = key_class_counts_train_repo
        key_task_counts = key_task_counts_train_repo
        dir += "train"
    elif args.dataset == "test":
        key_class_counts = key_class_counts_test_repo
        key_task_counts = key_task_counts_test_repo
        dir += "test"
    elif args.dataset == "training":
        key_class_counts = key_class_counts_training_repo
        key_task_counts = key_task_counts_training_repo
        dir += "training"  
elif args.model == "avalanche":
    dir = "avalanche/"
    if args.dataset == "train":
        key_class_counts = key_class_counts_train
        key_task_counts = key_task_counts_train
        dir += "train"
    elif args.dataset == "test":
        key_class_counts = key_class_counts_test
        key_task_counts = key_task_counts_test
        dir += "test"
    elif args.dataset == "training":
        key_class_counts = key_class_counts_training
        key_task_counts = key_task_counts_training
        dir += "training"
elif args.model == "random":
    dir = "random/"
    if args.dataset == "train":
        key_class_counts = key_class_counts_train_random
        key_task_counts = key_task_counts_train_random
        dir += "train"
    elif args.dataset == "test":
        key_class_counts = key_class_counts_test_random
        key_task_counts = key_task_counts_test_random
        dir += "test"
elif args.model == "repo_100":
    dir = "repo_100/"
    if args.dataset == "train":
        key_class_counts = key_class_counts_train_repo_100
        key_task_counts = key_task_counts_train_repo_100
        dir += "train"
    elif args.dataset == "test":
        key_class_counts = key_class_counts_test_repo_100
        key_task_counts = key_task_counts_test_repo_100
        dir += "test"


print(f"Saving plots in {plots_dir}/{dir}/")

print("Creating plots")
# with tqdm(total=2*num_keys+num_tasks+num_classes) as pbar:
#     for i in range(num_keys):
#         classes_counts = key_class_counts[i]
#         classes = classes_counts.keys()
#         counts = classes_counts.values()
#         plt.bar(classes, counts)
#         plt.xlabel("Classes")
#         plt.ylabel("Counts")
#         #plt.xticks("Classes")
#         plt.title(f"Key-class counts for key {i}")
#         plt.savefig(f'{plots_dir}/{dir}/key_class_counts/key_class_counts_key_{i}.png')
#         plt.clf()
#         pbar.update(1)

#     for i in range(num_keys):
#         tasks_counts = key_task_counts[i]
#         tasks = tasks_counts.keys()
#         counts = tasks_counts.values()
#         plt.bar(tasks, counts)
#         plt.xlabel("Tasks")
#         plt.ylabel("Counts")
#         #plt.xticks("Classes")
#         plt.title(f"Key-task counts for key {i}")
#         plt.savefig(f'{plots_dir}/{dir}/key_task_counts/key_task_counts_key_{i}.png')
#         plt.clf()
#         pbar.update(1)

    
#     task_key_counts = {i: {j: key_task_counts[j][i] for j in range(num_keys)} for i in range(num_tasks)}
#     for i in range(num_tasks):
#         keys_counts = task_key_counts[i]
#         keys = keys_counts.keys()
#         counts= keys_counts.values()
#         plt.bar(keys, counts)
#         plt.xlabel("Keys")
#         plt.ylabel("Counts")
#         #plt.xticks("Classes")
#         plt.title(f"Key-task counts for task {i}")
#         plt.savefig(f'{plots_dir}/{dir}/task_key_counts/task_key_counts_task_{i}.png')
#         plt.clf()
#         pbar.update(1)

#     class_key_counts = {i: {j: key_class_counts[j][i] for j in range(num_keys)} for i in range(num_classes)}
#     for i in range(num_classes):
#         keys_counts = class_key_counts[i]
#         keys = keys_counts.keys()
#         counts= keys_counts.values()
#         plt.bar(keys, counts)
#         plt.xlabel("Keys")
#         plt.ylabel("Counts")
#         #plt.xticks("Classes")
#         plt.title(f"Key-class counts for class {i}")
#         plt.savefig(f'{plots_dir}/{dir}/class_key_counts/class_key_counts_class_{i}.png')
#         plt.clf()
#         pbar.update(1)

if args.model != "repo_100":
    key_task_matrix = np.array([np.array([key_task_counts[i][j] for j in range(num_tasks)]) for i in range(num_keys)])
    plt.figure(figsize=(20, 20))
    hm = sns.heatmap(data=key_task_matrix, cmap="coolwarm", annot=True, square=True, annot_kws={"size": 20}, fmt='g')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("Tasks", size = 25)
    plt.ylabel("Keys", size = 25)
    plt.xticks(hm.get_xticks(), size= 20)
    plt.yticks(hm.get_yticks(), size= 20)
    plt.savefig(f'{plots_dir}/{dir}/heatmaps/key_task_heatmap.png')
    plt.clf()
    # plt.show()

    key_class_matrix = np.array([np.array([key_class_counts[i][j] for j in range(num_classes)]) for i in range(num_keys)])
    plt.figure(figsize=(200, 50))
    hm = sns.heatmap(data=key_class_matrix, cmap="coolwarm", annot=True, square=False, annot_kws={"fontsize": 50, 'rotation': 90}, fmt='g')
    plt.xlabel("Classes", size = 100, labelpad=30)
    plt.ylabel("Keys", size = 100,  labelpad=30)
    plt.yticks(hm.get_yticks(), size = 80)
    plt.xticks(hm.get_xticks(), size = 40)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=80)
    plt.savefig(f'{plots_dir}/{dir}/heatmaps/key_class_heatmap.png')
    plt.clf()
    # plt.show()
else:
    key_task_matrix = np.array([np.array([key_task_counts[i][j] for j in range(num_tasks)]) for i in range(num_keys)])
    plt.figure(figsize=(50, 200))
    hm = sns.heatmap(data=key_task_matrix, cmap="coolwarm", annot=True, square=False, annot_kws={"size": 20}, fmt='g')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("Tasks", size = 25)
    plt.ylabel("Keys", size = 25)
    plt.xticks(hm.get_xticks(), size= 20)
    plt.yticks(hm.get_yticks(), size= 20)
    plt.savefig(f'{plots_dir}/{dir}/heatmaps/key_task_heatmap.png')
    plt.clf()
    # plt.show()

    key_class_matrix = np.array([np.array([key_class_counts[i][j] for j in range(num_classes)]) for i in range(num_keys)])
    plt.figure(figsize=(80, 80))
    hm = sns.heatmap(data=key_class_matrix, cmap="coolwarm", annot=True, square=True, annot_kws={"fontsize": 10}, fmt='g')
    plt.xlabel("Classes", size = 100, labelpad=30)
    plt.ylabel("Keys", size = 100,  labelpad=30)
    plt.yticks(hm.get_yticks(), size = 10)
    plt.xticks(hm.get_xticks(), size = 10)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=80)
    plt.savefig(f'{plots_dir}/{dir}/heatmaps/key_class_heatmap.png')
    plt.clf()
    # plt.show()