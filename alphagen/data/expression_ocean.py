from .expression import (
    Operator,
    BinaryOperator,
    UnaryOperator,
    PairRollingOperator,
    RollingOperator,
)
import numpy as np
import torch
from torch import Tensor
import bottleneck as bk

N_PROD = 6000
FILL = "nofill"


def nanstd(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        torch.nanmean(
            (input_tensor - torch.nanmean(input_tensor, dim=1, keepdim=True)) ** 2
        ),
    )


def nanmax(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.max(input_tensor[~torch.isnan(input_tensor)], dim=1, keepdim=True)


def nanmin(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.min(input_tensor[~torch.isnan(input_tensor)], dim=1, keepdim=True)


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


class BFill(UnaryOperator):
    def _apply(self, x: Tensor) -> Tensor:
        x = torch.flip(x, [0])
        ffill = FFill()
        x = ffill._apply(x)
        x = torch.flip(x, [0])
        return x


def div(x, y) -> Tensor:
    return torch.randn(10)


class CrsStd(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand = torch.nan_to_num(operand, torch.nan, torch.nan, torch.nan)
        res = None  # TODO

        return None


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


class Roll(RollingOperator):
    def _apply(self, operand: Tensor, window: int) -> Tensor:
        return torch.nan_to_num(torch.roll(operand, shifts=window, dims=1), 0, 0, 0)
