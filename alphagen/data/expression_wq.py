from functools import wraps
from re import T
from typing import *

import bottleneck as bk
import numpy as np
import torch as th
from torch import Tensor

from .expression import (
    BinaryOperator,
    PairRollingOperator,
    RollingOperator,
    ShiftOperator,
    UnaryOperator,
    roll_with_nan,
)
from .expression_ocean import *

N_PROD = 6000
EPSILON = 1e-8


class Fraction(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return th.sign(operand) * (th.abs(operand) - th.floor(th.abs(operand)))


class SignedPower(ShiftOperator):
    def _apply(self, operand: Tensor, delta_time: int) -> Tensor:
        return th.float_power(operand, delta_time) * th.sign(operand)


class ArcTan(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return th.atan(operand)


def batch_linear_regression_ignore_nan(X, Y):
    mask = th.logical_and(th.isfinite(X), th.isfinite(Y))

    X_mean = th.sum(X * mask, dim=1, keepdim=True) / th.sum(mask, dim=1, keepdim=True)
    Y_mean = th.sum(Y * mask, dim=1, keepdim=True) / th.sum(mask, dim=1, keepdim=True)

    numerator = th.sum((X - X_mean) * (Y - Y_mean) * mask, dim=1, keepdim=True)
    denominator = th.sum((X - X_mean) ** 2 * mask, dim=1, keepdim=True)

    slope = numerator / denominator
    intercept = Y_mean - slope * X_mean

    intercept_repeated = intercept.repeat(1, 6000)
    slope = slope.repeat(1, 6000)
    return slope, intercept_repeated


class CrsRegCoef(BinaryOperator):
    def _apply(self, x: Tensor, y: Tensor) -> Tensor:
        _, intercept_repeated = batch_linear_regression_ignore_nan(x, y)
        return intercept_repeated


class CrsRegResidual(BinaryOperator):
    def _apply(self, x: Tensor, y: Tensor) -> Tensor:
        slope, _ = batch_linear_regression_ignore_nan(x, y)
        return slope


class TsRegCoef(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        mean_lhs = lhs.nanmean(dim=-1, keepdim=True)
        mean_rhs = rhs.nanmean(dim=-1, keepdim=True)
        clhs = lhs - mean_lhs
        crhs = rhs - mean_rhs
        numerator = (clhs * crhs).sum(dim=-1)
        denominator = (clhs**2).sum(dim=-1)
        slope = numerator / denominator
        intercept = mean_rhs.squeeze(-1) - slope * mean_lhs.squeeze(-1)

        return intercept


class CoSkew(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        mean_lhs = lhs.nanmean(dim=-1, keepdim=True)
        mean_rhs = rhs.nanmean(dim=-1, keepdim=True)
        clhs = lhs - mean_lhs
        crhs = rhs - mean_rhs
        var_lhs = (clhs**2).sum(dim=-1)
        var_rhs = (crhs).sum(dim=-1)
        coskewness = ((clhs**2) * crhs).sum(dim=-1)
        std_lhs = torch.sqrt(var_lhs)
        std_rhs = torch.sqrt(var_rhs)
        standardized_coskewness = coskewness / (std_lhs**2 * std_rhs)
        return standardized_coskewness


class CoKurtosis(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        mean_lhs = lhs.nanmean(dim=-1, keepdim=True)
        mean_rhs = rhs.nanmean(dim=-1, keepdim=True)
        clhs = lhs - mean_lhs
        crhs = rhs - mean_rhs
        ncentral_moment = ((clhs**2) * (crhs**2)).sum(dim=-1)
        var_lhs = (clhs**3).sum(dim=-1)
        var_rhs = (crhs).sum(dim=-1)
        std_lhs = torch.sqrt(var_lhs)
        std_rhs = torch.sqrt(var_rhs)
        stdmul = std_lhs**3 * std_rhs
        stdmul[(var_lhs < 1e-6) | (var_rhs < 1e-6)] = 1
        return ncentral_moment / (stdmul)


Operators_wq = [
    Fraction,
    SignedPower,
    ArcTan,
    CrsRegCoef,
    CrsRegResidual,
    TsRegCoef,
    CoSkew,
    CoKurtosis,
]
