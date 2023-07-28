from typing import List
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.stock_data_ import *
import numpy as np


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData):
        self.data = data
        self.ret1d = np.load(
            "/home/public2/share_yw/data/basic_info/RET1D.npy"
        ).reshape(-1, N_PROD)[data.start_idx : data.end_idx]
        self.ret2d = np.load(
            "/home/public2/share_yw/data/basic_info/RET2D.npy"
        ).reshape(-1, N_PROD)[data.start_idx : data.end_idx]
        self.ret5d = np.load(
            "/home/public2/share_yw/data/basic_info/RET2D.npy"
        ).reshape(-1, N_PROD)[data.start_idx : data.end_idx]
        self.ret1d = torch.from_numpy(self.ret1d)
        print(f"ret1d shape:{self.ret1d.shape}")

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def _make_ensemble_alpha(
        self, exprs: List[Expression], weights: List[float]
    ) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [
            self._calc_alpha(exprs[i]) * weights[i] for i in range(n)
        ]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.ret1d)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.ret1d).mean().item()
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.ret1d).mean().item()
            return rank_ic
