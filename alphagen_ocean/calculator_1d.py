from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch import Tensor

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.dir_config import DIR_RETS
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, pool_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.stock_data import N_PROD, ArgData


class Calculator1d(AlphaCalculator):
    def __init__(
        self,
        data: ArgData,
        device=torch.device("cpu"),
    ):
        self.data = data
        self.device = device

        real_start_idx = data.start_idx + data.max_backtrack_days
        real_end_idx = data.end_idx - data.max_future_days
        self.raw_ret1d = np.load(DIR_RETS[0]).reshape(-1, N_PROD)
        self.ret1d = self.raw_ret1d[real_start_idx:real_end_idx]
        self.ret1d = torch.from_numpy(self.ret1d).to(device)

    @lru_cache(maxsize=50)
    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    @lru_cache(maxsize=50)
    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    @lru_cache(maxsize=50)
    def _calc_pIC(self, signal: Tensor, target: Tensor) -> float:
        return pool_pearsonr(signal, target).item()

    @lru_cache(maxsize=50)
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

    @lru_cache(maxsize=50)
    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)

        ic_1d = self._calc_IC(value, self.ret1d)
        return ic_1d

    @lru_cache(maxsize=50)
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.ret1d).mean().item()
            return ic

    def calc_pool_pIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            pic = pool_pearsonr(ensemble_value, self.ret1d).item()
            return pic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.ret1d).mean().item()
            return rank_ic

    def _calc_q90(self, value: Tensor, target: Tensor) -> float:
        top_10_percent_idx = int(0.1 * len(value))
        sorted_indices = torch.argsort(value, descending=True)
        top_10_percent_values = target[sorted_indices[:top_10_percent_idx]]
        return top_10_percent_values.mean().item()

    def calc_q90_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_q90(value, self.ret1d)
