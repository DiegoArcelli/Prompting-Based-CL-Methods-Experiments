import sys
sys.path.append("./../")
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import key_class_counts, key_task_counts
import seaborn as sns
import numpy as np

num_keys = 10
num_classes = 100
num_tasks = 10

print("Creating bar plots")
with tqdm(total=2*num_keys+num_tasks+num_classes) as pbar:
    for i in range(num_keys):
        classes_counts = key_class_counts[i]
        classes = classes_counts.keys()
        counts = classes_counts.values()
        plt.bar(classes, counts)
        plt.xlabel("Classes")
        plt.ylabel("Counts")
        #plt.xticks("Classes")
        plt.title(f"Key-class counts for key {i}")
        plt.savefig(f'../../plots//key_class_counts/key_class_counts_key_{i}.png')
        plt.clf()
        pbar.update(1)

    for i in range(num_keys):
        tasks_counts = key_task_counts[i]
        tasks = tasks_counts.keys()
        counts = tasks_counts.values()
        plt.bar(tasks, counts)
        plt.xlabel("Tasks")
        plt.ylabel("Counts")
        #plt.xticks("Classes")
        plt.title(f"Key-task counts for key {i}")
        plt.savefig(f'../../plots/key_task_counts/key_task_counts_key_{i}.png')
        plt.clf()
        pbar.update(1)

    
    task_key_counts = {i: {j: key_task_counts[j][i] for j in range(num_keys)} for i in range(num_tasks)}
    for i in range(num_tasks):
        keys_counts = task_key_counts[i]
        keys = keys_counts.keys()
        counts= keys_counts.values()
        plt.bar(keys, counts)
        plt.xlabel("Keys")
        plt.ylabel("Counts")
        #plt.xticks("Classes")
        plt.title(f"Key-task counts for task {i}")
        plt.savefig(f'../../plots/task_key_counts/task_key_counts_task_{i}.png')
        plt.clf()
        pbar.update(1)

    class_key_counts = {i: {j: key_class_counts[j][i] for j in range(num_keys)} for i in range(num_classes)}
    for i in range(num_classes):
        keys_counts = class_key_counts[i]
        keys = keys_counts.keys()
        counts= keys_counts.values()
        plt.bar(keys, counts)
        plt.xlabel("Keys")
        plt.ylabel("Counts")
        #plt.xticks("Classes")
        plt.title(f"Key-class counts for class {i}")
        plt.savefig(f'../../plots/class_key_counts/class_key_counts_class_{i}.png')
        plt.clf()
        pbar.update(1)

key_task_matrix = np.array([np.array([key_task_counts[i][j] for j in range(num_tasks)]) for i in range(num_keys)])
hm = sns.heatmap(data=key_task_matrix, annot=False, square=True)
plt.xlabel("Tasks")
plt.ylabel("Keys")
plt.savefig(f'../../plots/heatmaps/key_task_heatmap.png')
plt.clf()
# plt.show()


key_class_matrix = np.array([np.array([key_class_counts[i][j] for j in range(num_classes)]) for i in range(num_keys)])
hm = sns.heatmap(data=key_class_matrix, annot=False, square=True)
plt.xlabel("Classes")
plt.ylabel("Keys")
plt.savefig(f'../../plots/heatmaps/key_class_heatmap.png')
plt.clf()
# plt.show()

