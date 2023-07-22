import torch

def prompt_forward_features(model, x, prompt_id, cls_features=None):
    x = x.float()
    x = model.patch_embed(x)

    batch_size = x.shape[0]
    res = {}

    # selected_prompt = model.prompt.prompt[[prompt_id for _ in range(model.prompt.top_k)], : ,:]
    # selected_prompt = selected_prompt.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    selected_prompt = model.prompt.prompt[prompt_id, : ,:]
    selected_prompt = selected_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
    res["prompted_embedding"] = torch.cat([selected_prompt, x], dim=1)
    model.total_prompt_len = selected_prompt.shape[1]
    x = res["prompted_embedding"]

    if model.cls_token is not None:
        x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed[:, :-20, :])
    x = model.blocks(x)
    x = model.norm(x)
    res["x"] = x

    return res


def prompt_forward(model, x, prompt_id):
    res = prompt_forward_features(model, x, prompt_id)
    res = model.forward_head(res)
    return res


def select_classes(dataset, class_id):
    pass