from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

Tensor = Union[torch.Tensor]


def masked_mean_std(
    x: Tensor, n: Optional[Tensor] = None, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    `x`: [days, stocks], input data
    `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid necessary computations
    `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
    defaults to `torch.isnan(x)`..
    """
    if mask is None:
        mask = torch.isnan(x)
    if n is None:
        n = (~mask).sum(dim=1)

    # Precompute zeros_like(x) to avoid multiple computations
    zeros = torch.zeros_like(x)

    # Use precomputed zeros
    masked_x = torch.where(mask, zeros, x)
    mean = masked_x.sum(dim=1) / n

    # Compute the variance using precomputed zeros and mean
    variance = torch.where(mask, zeros, (x - mean.unsqueeze(-1)) ** 2).sum(dim=1) / n

    # Compute standard deviation
    std = torch.sqrt(variance)

    return mean, std


def normalize_by_day(value: Tensor) -> Tensor:
    mean, std = masked_mean_std(value)
    value.sub_(mean[:, None]).div_(std[:, None])
    nan_mask = torch.isnan(value)
    value.masked_fill_(nan_mask, 0.0)
    return value


def normalize_timeseries(value: Tensor) -> Tensor:
    value = value.nan_to_num(0, 0, 0)
    mean = value.mean(dim=0)
    std = value.std(dim=0)
    value.sub_(mean).div_(std + 1)
    return value


# def masked_mean_std(
#     x: Tensor, n: Optional[Tensor] = None, mask: Optional[Tensor] = None
# ) -> Tuple[Tensor, Tensor]:
#     """
#     `x`: [days, stocks], input data
#     `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid necessary computations
#     `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
#     defaults to `torch.isnan(x)`
#     """
#     if mask is None:
#         mask = torch.isnan(x)
#     if n is None:
#         n = (~mask).sum(dim=1)
#     x = x.clone()
#     x[mask] = 0.0
#     mean = x.sum(dim=1) / n
#     std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
#     return mean, std


# def normalize_by_day(value: Tensor) -> Tensor:
#     mean, std = masked_mean_std(value)
#     value = (value - mean[:, None]) / std[:, None]
#     nan_mask = torch.isnan(value)
#     value[nan_mask] = 0.0
#     return value


def scale_to_neg_one_to_one(tensor):
    """
    将 PyTorch Tensor 按比例缩放到 -1 到 1 之间，并进行检查。

    参数:
    tensor (torch.Tensor): 要缩放的输入张量

    返回:
    torch.Tensor: 缩放到 -1 到 1 之间的张量
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    min_val = tensor.min()
    max_val = tensor.max()

    # 检查是否存在单一值的情况
    if min_val == max_val:
        # 如果所有元素都相同，返回一个全0的张量
        return torch.zeros_like(tensor)

    # Scale to [0, 1]
    scaled_tensor = (tensor - min_val) / (
        max_val - min_val + 1e-8
    )  # 添加一个很小的值以避免除以0

    # Scale to [-1, 1]
    scaled_tensor = 2 * scaled_tensor - 1

    # 检查结果是否在 -1 到 1 之间
    if scaled_tensor.min() < -1 or scaled_tensor.max() > 1:
        raise ValueError(
            "Scaling failed. The result tensor has values outside the range [-1, 1]."
        )

    return scaled_tensor


def standardize_tensor(tensor, threshold=1e-5):
    """
    将 PyTorch Tensor 标准化，使得结果具有均值 0 和标准差 1。
    如果原始标准差非常小（小于给定的阈值），则使用 1 作为标准差来避免异常放大的问题。

    参数:
    tensor (torch.Tensor): 要标准化的输入张量
    threshold (float): 判断标准差是否足够大的阈值

    返回:
    torch.Tensor: 标准化后的张量
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    mean_val = tensor.mean()
    std_val = tensor.std()

    # 如果标准差太小，为了避免异常放大的问题，使用 1 作为标准差
    if std_val < threshold:
        std_val = 1

    # 标准化操作
    standardized_tensor = (tensor - mean_val) / std_val

    return standardized_tensor
