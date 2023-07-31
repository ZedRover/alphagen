from .expression import (
    BinaryOperator,
    UnaryOperator,
    # PairRollingOperator,
    RollingOperator,
)
import numpy as np
import torch
from torch import Tensor
import bottleneck as bk
from functools import wraps


N_PROD = 6000
FILL = "nofill"
EPSILON = 1e-8


def fillnan(x: Tensor) -> Tensor:
    if FILL == "zerofill":
        return torch.nan_to_num(x, 0, 0, 0)
    elif FILL == "crsmeanfill":
        return x
    elif FILL == "nofill":
        return x
    elif FILL == "ffill":
        return x
    elif FILL == "bfill":
        return x
    else:
        return x


def fillinf(func):
    @wraps(func)
    def fillfunc(*args, **kwargs):
        mask = torch.tensor(
            [torch.isfinite(arg) for arg in args if isinstance(arg, Tensor)]
        ).all(dim=0)

        x = func(*args, **kwargs)
        if x.dtype == torch.float32 or x.dtype == torch.float64:
            x[torch.isinf(x)] = torch.tensor(float("nan"))
        x[~mask] = torch.tensor(float("nan"))
        return x

    return fillfunc


def div(x: Tensor, y: Tensor) -> Tensor:
    y = torch.where(torch.abs(y) < EPSILON, 0, y)
    x = torch.where(torch.abs(x) < EPSILON, 0, x)
    res = fillnan(torch.divide(x, y))
    res = torch.where(torch.isinf(res), torch.nan, res)
    return fillnan(res)


def nanstd(
    input_tensor: torch.Tensor, dim: int = 0, keepdim: bool = True
) -> torch.Tensor:
    return torch.sqrt(
        torch.nanmean(
            (input_tensor - torch.nanmean(input_tensor, dim=dim, keepdim=keepdim)) ** 2
        ),
    )


def nanmax(input_tensor: torch.Tensor, dim=1, keepdim=True) -> torch.Tensor:
    return torch.max(input_tensor[~torch.isnan(input_tensor)], dim=dim, keepdim=keepdim)


def nanmin(input_tensor: torch.Tensor, dim=1, keepdim=True) -> torch.Tensor:
    return torch.min(input_tensor[~torch.isnan(input_tensor)], dim=dim, keepdim=keepdim)


class CrsRank(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        nx = x.detach().cpu().numpy()
        validnum = np.isfinite(nx).sum(axis=1, keepdims=True)
        nx = bk.nanrankdata(nx, axis=1) - (validnum + 1) / 2
        return torch.from_numpy(nx).to(x.device)


class BiasCrsRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nx = operand.detach().cpu().numpy()
        nx = bk.nanrankdata(nx, axis=1)
        return torch.from_numpy(nx).to(operand.device)


class NormCrsRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nx = operand.detach().cpu().numpy()
        validnum = np.isfinite(nx).sum(axis=1, keepdims=True)
        nx = bk.nanrankdata(nx, axis=1) / validnum - 0.5
        return torch.from_numpy(nx).to(operand.device)


class FFillBase(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand = operand.t()
        mask = torch.isfinite(operand)
        idx = torch.where(mask, torch.arange(mask.shape[1]), 0)
        torch.maximum.accumulate(idx, dim=1, out=idx)
        out = operand[idx, torch.arange(idx.shape[0])[None, :]].t()
        out = torch.nan_to_num(out, 0, 0, 0)
        return out


class FFill(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        mask = torch.isfinite(x)
        idx = torch.where(mask, torch.arange(mask.shape[0])[:, None], 0)
        torch.maximum.accumulate(idx, dim=0, out=idx)
        out = x[idx, torch.arange(idx.shape[1])[None, :]]
        out = torch.nan_to_num(out, 0, 0, 0)
        return out


def _ffill(x):
    mask = torch.isfinite(x)
    idx = torch.where(mask, torch.arange(mask.shape[0])[:, None], 0)
    torch.maximum.accumulate(idx, dim=0, out=idx)
    out = x[idx, torch.arange(idx.shape[1])[None, :]]
    out = torch.nan_to_num(out, 0, 0, 0)
    return out


class BFill(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        x = torch.flip(x, [0])
        x = _ffill(x)
        x = torch.flip(x, [0])
        return x


class CrsStd(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand = torch.nan_to_num(operand, torch.nan, torch.nan, torch.nan)
        res = div(
            operand - torch.nanmean(operand, dim=1, keepdim=True), nanstd(operand)
        )
        return fillnan(res)


class CapCrsStd(RollingOperator):
    def _apply(self, operand: Tensor, window: int) -> Tensor:
        operand = operand - torch.nanmean(operand, dim=1, keepdim=True)
        std = nanstd(operand, dim=1, keepdim=True)
        mask = torch.abs(operand) < window * std
        sign = torch.sign(operand)
        operand = torch.where(mask, operand, torch.nan)
        operand = div(
            operand - torch.nanmean(operand, dim=1, keepdim=True),
            nanstd(operand),
        )
        operand = torch.where(
            mask,
            operand,
            (sign > 0) * nanmax(operand) + (sign < 0) * nanmin(operand),
        )
        return operand


class Cap(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.clip(operand, -10, 10)


class CrsAbs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = (operand - torch.nanmean(operand, dim=1, keepdim=True)) / torch.nansum(
            torch.abs(operand), dim=1, keepdim=True
        )
        return torch.nan_to_num(res)


class DeStd(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = div(operand, nanstd(operand))
        return torch.nan_to_num(res, 0, 0, 0)


class DeNorm(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = div(operand, torch.sqrt(torch.nanmean(operand**2, dim=1, keepdim=True)))
        return torch.nan_to_num(res, 0, 0, 0)


class CrsDemean(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = operand - torch.nanmean(operand, dim=1, keepdim=True)
        return torch.nan_to_num(res, 0, 0, 0)


class CrsMean(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand = operand - torch.nanmean(operand, dim=1, keepdim=True)
        operand = torch.nan_to_num(operand, 0, 0, 0)
        return operand


class RollNp(RollingOperator):
    def _apply(self, operand: Tensor, window: int) -> Tensor:
        rolled = torch.roll(operand, shifts=window, dims=1)
        rolled[:, :window] = torch.nan
        return rolled


class Roll(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.nan_to_num(torch.roll(operand, shifts=1, dims=0), 0, 0, 0)


class TickNanRoll(RollingOperator):
    def _apply(self, operand: Tensor, window: int) -> Tensor:
        operand = torch.roll(operand, shifts=window, dims=0)
        operand[:window] = torch.nan
        return torch.nan_to_num(operand, 0, 0, 0)


class Delta(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand - torch.roll(operand, shifts=1, dims=0)


class TickDelta(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        rolled = torch.roll(operand, shifts=window, dims=0)
        rolled[:window] = torch.nan
        return operand - rolled


def _tick_delta(x: Tensor, window: int = 1) -> Tensor:
    rolled = torch.roll(x, shifts=window, dims=0)
    rolled[:window] = torch.nan
    return x - rolled


class TickDDRatio(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return div(_tick_delta(lhs), _tick_delta(rhs)) - 1


class TickLinearDecay(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        rolled = torch.roll(operand, shifts=window, dims=0)
        rolled[:window] = torch.nan
        return operand - rolled


# TimeSeries Funcs


class TsMax(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return nanmax(res, dim=1, keepdim=True)


class TsMin(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return nanmin(res, dim=1, keepdim=True)


class TsStd(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return nanstd(res, dim=1, keepdim=True)


class TsMean(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return torch.nanmean(res, dim=1, keepdim=True)


class TsSum(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return torch.nansum(res, dim=1, keepdim=True)


class TsRank(RollingOperator):
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        mask = torch.isnan(res)
        validnum = torch.sum(torch.isfinite(res), dim=1, keepdim=True)
        low = torch.sum(res < res[[0], :], dim=1, keepdim=True)
        high = validnum - torch.sum(res > res[[0], :], dim=1, keepdim=True)
        rank = (high + low - 1) / (validnum - 1) - 1
        rank[validnum == 1] = 0
        rank[mask[0]] = torch.tensor(float("nan"))
        return rank


class TickTsMeanDiff(RollingOperator):
    @fillinf
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        mean = torch.nanmean(res, dim=1)
        return operand - mean


class TickTsRankSingle(RollingOperator):
    @fillinf
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        rank = torch.argsort(torch.argsort(res, dim=1), dim=1)
        return window / 2 * rank


class TickTsMaxSingle(RollingOperator):
    @fillinf
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return nanmax(res, dim=1)


class TickTsMinSingle(RollingOperator):
    @fillinf
    def _apply(self, operand: Tensor, window: int = 5) -> Tensor:
        res = operand.unfold(0, window, 1)
        return nanmin(res, dim=1)


# OCOperators = [
#     CrsRank,
#     BiasCrsRank,
#     NormCrsRank,
#     FFillBase,
#     FFill,
#     BFill,
#     CrsStd,
#     CapCrsStd,
#     Cap,
#     CrsAbs,
#     DeStd,
#     DeNorm,
#     CrsDemean,
#     CrsMean,
#     RollNp,
#     Roll,
#     TickNanRoll,
#     Delta,
#     TickDelta,
#     TickDDRatio,
#     TickLinearDecay,
# ]
