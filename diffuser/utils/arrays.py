# import collections
import numpy as np
import torch
# import pdb

# -----------------------------------------------------------------------------#
# ------------------------------ numpy <--> torch -----------------------------#
# -----------------------------------------------------------------------------#


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=torch.float, device='cpu'):
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        raise RuntimeError(f"Unrecognized type in `to_device`: {type(x)}")


def batchify(batch, device):
    """
    convert a single dataset item to a batch suitable for passing to a model by
            1) converting np arrays to torch tensors and
            2) and ensuring that everything has a batch dimension
    """
    fn = lambda x: to_torch(x[None], device=device)

    batched_vals = []
    for field in batch._fields:
        val = getattr(batch, field)
        val = apply_dict(fn, val) if type(val) is dict else fn(val)
        batched_vals.append(val)
    return type(batch)(*batched_vals)


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x = x - x.min()
    x = x / x.max()
    return x


def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255).astype(np.uint8)


# def set_device(device):
#     if "cuda" in device:
#         torch.set_default_tensor_type(torch.cuda.FloatTensor)


def batch_to_device(batch, device):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name

# 报告给定模型参数的统计信息。
def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    #计算模型中的总参数数量。
    n_parameters = sum(counts.values())
    # 打印总的参数数量。
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")
    # 取模型所有模块（及其子模块）的字典。
    modules = dict(model.named_modules())
    # 将模型中的模块按参数数量降序排序。
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    # 对参数数量最多的 topk 个模块进行遍历。
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    # 打印当前模块的名称、参数数量和模块内容。
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters
