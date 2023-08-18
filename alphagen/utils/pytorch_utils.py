from typing import Tuple, Optional, Union
import torch
from torch import nn, Tensor

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

    masked_x = torch.where(mask, torch.zeros_like(x), x)
    mean = masked_x.sum(dim=1) / n

    variance = (
        torch.where(mask, torch.zeros_like(x), (x - mean.unsqueeze(-1)) ** 2).sum(dim=1)
    ) / n
    std = torch.sqrt(variance)

    return mean, std


def normalize_by_day(value: Tensor) -> Tensor:
    mean, std = masked_mean_std(value)
    value.sub_(mean[:, None]).div_(std[:, None])
    value = torch.where(torch.isnan(value), torch.zeros_like(value), value)
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
