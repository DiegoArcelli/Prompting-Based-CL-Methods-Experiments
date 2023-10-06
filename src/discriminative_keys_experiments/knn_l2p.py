import torch
# from prompt_selection_experiments.l2p import create_model
from avalanche.models.vit import create_model
import numpy as np
from functools import reduce

class KNNLearningToPrompt:

    def __init__(
        self, 
        model_path: str = "./../../checkpoints/l2p_cifar100_l2p_selection.pt",
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 100,
        num_tasks: int = 10,
        predict_task: bool = True,
        device: str = "cpu",
        k: int = 5
    ):

        self.model = torch.load(model_path).to(device)
        self.num_classes = num_classes

        self.original_vit = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
        ).to(device)
        
        self.original_vit.reset_classifier(0)
        for p in self.original_vit.parameters():
            p.requires_grad = False 

        self.pool_size = self.model.prompt.pool_size
        self.num_tasks = num_tasks
        self.top_k = self.model.prompt.top_k
        self.predict_task = predict_task
        self.k = k
        self.device = device

        # for each prompt we count how many time it is selected for each class
        self.key_class_counts = {k: {c: 0 for c in range(self.num_classes)} for k in range(self.pool_size)}   
        # for each prompt we count how many time it is selected for each task
        self.key_task_counts = {k: {t: 0 for t in range(self.num_tasks)} for k in range(self.pool_size)}

    def train(self, x: torch.Tensor, labels: torch.Tensor):

        with torch.no_grad():
            cls_features = self.original_vit(x)["pre_logits"]

        res = self.model(
            x=x,
            task_id=-1,
            cls_features=cls_features,
            train=False,
        )

        # logits = res["logits"]

        # iterate over each prediction
        batch_size = labels.shape[0]
        for i in range(batch_size):
            label = labels[i].item() # get the true class of the i-th element of the batch
            # iterate over each prompt selected prompt for the i-th element of the batch
            for j in range(self.top_k): 
                key_id = res["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                # increase the key-class counter
                task = label // self.num_tasks
                self.key_class_counts[key_id][label] += 1
                self.key_task_counts[key_id][task] += 1


    def compute_key_class_mapping(self):
        self.key_class_mapping = {key: reduce(lambda agg, x: agg if agg[1] > x[1] else x, list(counts.items()), (0,0))[0] for key, counts in self.key_class_counts.items()}
        self.key_task_mapping = {key: reduce(lambda agg, x: agg if agg[1] > x[1] else x, list(counts.items()), (0,0))[0] for key, counts in self.key_task_counts.items()}


    def predict(self, x: torch.Tensor):

        with torch.no_grad():
            cls_features = self.original_vit(x)["pre_logits"]

        # normalize the key and the query
        keys = self.model.prompt.l2_normalize(self.model.prompt.prompt_key, dim=1) # (N_K, D)
        query = self.model.prompt.l2_normalize(cls_features, dim=1) # (B, D)

        # compute the cosine similarity between the queries and the keys
        similarity = torch.matmul(query, keys.T) # (B, N_K)

        # find the top k keys for each element of the batch
        _, idx = torch.topk(similarity, k=self.k, dim=1) # (B, K)

        # we map each one of the top k keys to its correspondent class   
        if self.predict_task:
            keys_preds = idx.detach().cpu().apply_(lambda x: self.key_task_mapping[x]).to(self.device) # (B, K)
        else:
            keys_preds = idx.detach().cpu().apply_(lambda x: self.key_class_mapping[x]).to(self.device) # (B, K)
        # classes = idx.detach().cpu().apply_(lambda x: key_class_map[x]).to(self.device) # (B, K)

        # compute the most present classes for each batch element
        pred = keys_preds.max(dim=1)[0] # (B)

        return pred