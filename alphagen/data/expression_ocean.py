from .expression import (
    BinaryOperator,
    PairRollingOperator,
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


def shape_checking(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        input_shapes = [arg.shape for arg in args if isinstance(arg, torch.Tensor)]
        result = func(*args, **kwargs)
        output_shape = result.shape
        if input_shapes != output_shape:
            print(
                f"Function {func.__name__} changes shape from {input_shapes} to {output_shape}"
            )
        return result

    return wrapper


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
    input_tensor = input_tensor.clone()
    input_tensor[torch.isnan(input_tensor)] = float("-inf")
    max_values, _ = torch.max(input_tensor, dim=dim, keepdim=keepdim)
    return max_values


def nanmin(input_tensor: torch.Tensor, dim=1, keepdim=True) -> torch.Tensor:
    input_tensor = input_tensor.clone()
    input_tensor[torch.isnan(input_tensor)] = float("inf")
    return torch.min(input_tensor, dim=dim, keepdim=keepdim)


def nanrankdata(tensor, axis=0):
    tensor = tensor.clone()
    mask = torch.isnan(tensor)
    tensor[mask] = -1
    _, indices = tensor.sort(dim=axis)
    ranks = torch.zeros_like(tensor)
    ranks.scatter_(
        dim=axis,
        index=indices,
        src=torch.arange(1, tensor.shape[axis] + 1, dtype=tensor.dtype)
        .expand_as(tensor)
        .to(tensor.device),
    )
    ranks[mask] = float("nan")
    return ranks


class Square(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.nan_to_num(operand**2, 0, 0, 0)


class Sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sign(operand) * torch.sqrt(torch.abs(operand))


class Deltaratio(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.nan_to_num(div(lhs, rhs), 0, 0, 0)


class NormDeltaratio(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.nan_to_num(div(lhs - rhs, lhs + rhs), 0, 0, 0)


class CrsRank(UnaryOperator):
    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        validnum = (~torch.isnan(x)).sum(dim=1, keepdims=True)  # type: ignore
        nx = nanrankdata(x, axis=1) - (validnum + 1) / 2
        return nx


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
        idx = torch.cummax(idx, dim=1).values  # use torch.cummax here
        out = operand[idx, torch.arange(idx.shape[0])[None, :]].t()
        out = torch.nan_to_num(out, 0, 0, 0)
        return out


class FFill(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        mask = torch.isfinite(x)
        idx = torch.where(mask, torch.arange(mask.shape[0])[:, None], 0)
        idx = torch.cummax(idx, dim=0).values  # use torch.cummax here
        out = x[idx, torch.arange(idx.shape[1])[None, :]]
        out = torch.nan_to_num(out, 0, 0, 0)
        return out


def _ffill(x):
    mask = torch.isfinite(x)
    idx = torch.where(mask, torch.arange(mask.shape[0])[:, None], 0)
    idx = torch.cummax(idx, dim=0).values  # use torch.cummax here
    out = x[idx, torch.arange(idx.shape[1])[None, :]]
    out = torch.nan_to_num(out, 0, 0, 0)
    return out


class BFill(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        x = torch.flip(x, [0])
        x = _ffill(x)
        x = torch.flip(x, [0])
        return x


class CrsStd(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand = torch.nan_to_num(operand, torch.nan, torch.nan, torch.nan)
        res = div(
            operand - torch.nanmean(operand, dim=1, keepdim=True), nanstd(operand)
        )
        return fillnan(res)


class CapCrsStd(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        window = 3
        operand = operand - torch.nanmean(operand, dim=-1, keepdim=True)
        std = nanstd(operand, dim=-1, keepdim=True)
        mask = torch.abs(operand) < window * std
        sign = torch.sign(operand)
        operand = torch.where(mask, operand, torch.nan)
        operand = div(
            operand - torch.nanmean(operand, dim=-1, keepdim=True),
            nanstd(operand, dim=-1),
        )
        operand_max = torch.where(
            sign > 0, operand, torch.tensor(float("-inf")).to(operand.device)
        )
        operand_min = torch.where(
            sign < 0, operand, torch.tensor(float("inf")).to(operand.device)
        )
        operand = torch.where(
            mask, operand, torch.where(sign > 0, operand_max, operand_min)
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
        res = div(operand, nanstd(operand, dim=1))
        return torch.nan_to_num(res, 0, 0, 0)


class DeNorm(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = div(operand, torch.sqrt(torch.nanmean(operand**2, dim=1, keepdim=True)))
        return torch.nan_to_num(res, 0, 0, 0)


class CrsDeMean(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = operand - torch.nanmean(operand, dim=1, keepdim=True)
        return torch.nan_to_num(res, 0, 0, 0)


# class RollNp(RollingOperator):
#     def _apply(self, operand: Tensor, window: int) -> Tensor:
#         rolled = torch.roll(operand, shifts=window, dims=0)
#         rolled[:, :window] = torch.nan
#         return rolled


# class Roll(RollingOperator):
#     def _apply(self, operand: Tensor, window: int) -> Tensor:
#         rolled = torch.roll(operand, shifts=window, dims=0)
#         rolled[:, :window] = torch.nan
#         rolled = torch.nan_to_num(rolled, 0, 0, 0)
#         return rolled


# TimeSeries Funcs


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


def cossimfunc(stack_x, stack_y, dim=-1):
    xydot = (stack_x * stack_y).sum(dim=dim)
    xnorm = torch.linalg.norm(stack_x, 2, dim=dim)
    ynorm = torch.linalg.norm(stack_y, 2, dim=dim)

    return div(xydot, (xnorm * ynorm))


class CossImIc(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        res = cossimfunc(lhs, rhs, dim=-1)
        return res


class MinMaxStd(UnaryOperator):
    def _apply(self, operand: Tensor):
        operand_max = torch.nanquantile(operand, 0.99, dim=1, keepdim=True)
        operand_min = torch.nanquantile(operand, 0.01, dim=1, keepdim=True)
        operand = div(operand - operand_min, operand_max - operand_min)
        return torch.nan_to_num(operand, 0, 0, 0)


OCOperators = [
    Square,
    CrsRank,
    BiasCrsRank,
    NormCrsRank,
    FFill,
    BFill,
    CrsStd,
    # CapCrsStd,
    Cap,
    CrsAbs,
    DeStd,
    DeNorm,
    CrsDeMean,
]
