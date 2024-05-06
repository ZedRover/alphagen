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
    value.sub_(mean).div_(std)
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
