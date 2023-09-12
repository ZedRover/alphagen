import torch
from audtorch.metrics.functional import pearsonr
from torch import Tensor

from alphagen.utils.pytorch_utils import masked_mean_std


def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()  # [days, stocks]
    y = y.clone()  # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    rank = x.argsort().argsort().float()  # [d, s]
    eq = x[:, None] == x[:, :, None]  # [d, s, s]
    eq = eq / eq.sum(dim=2, keepdim=True)  # [d, s, s]
    rank = (eq @ rank[:, :, None]).squeeze(dim=2)
    rank[nan_mask] = 0
    return rank  # [d, s]


def _batch_pearsonr_given_mask(x: Tensor, y: Tensor, n: Tensor, mask: Tensor) -> Tensor:
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    return corrs


def timer(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} cost {round(end - start,2)} seconds")
        return ret

    return wrapper


# @timer
def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


# @timer
def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.0))


# @timer
def pool_pearsonr(signal: Tensor, target: Tensor) -> Tensor:
    signal = signal.reshape(-1, 1)
    target = target.reshape(-1, 1)
    nan_mask = torch.isinf(target) & torch.isinf(signal)
    signal, target = signal[nan_mask], target[nan_mask]
    print("signal:", signal)
    print("target:", target)
    return pearsonr(signal, target, batch_first=False)
