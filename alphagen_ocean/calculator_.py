from typing import List
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.stock_data_ import StockData, N_PROD
import numpy as np


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, device: torch.device = torch.device("cpu")):
        self.data = data
        self.device = device

        real_start_idx = data.start_idx + data.max_backtrack_days
        real_end_idx = data.end_idx - data.max_future_days

        self.ret1d = np.load(
            "/home/public2/share_yw/data/basic_info/RET1D.npy"
        ).reshape(-1, N_PROD)[real_start_idx:real_end_idx]
        self.ret2d = np.load(
            "/home/public2/share_yw/data/basic_info/RET2D.npy"
        ).reshape(-1, N_PROD)[real_start_idx:real_end_idx]
        self.ret5d = np.load(
            "/home/public2/share_yw/data/basic_info/RET5D.npy"
        ).reshape(-1, N_PROD)[real_start_idx:real_end_idx]
        self.ret1d = torch.from_numpy(self.ret1d).to(device)
        self.ret2d = torch.from_numpy(self.ret2d).to(device)
        self.ret5d = torch.from_numpy(self.ret5d).to(device)

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

        ic1 = self._calc_IC(value, self.ret1d)
        ic2 = self._calc_IC(value, self.ret2d)
        ic3 = self._calc_IC(value, self.ret5d)
        return (ic1 + ic2 + ic3) / 3

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.ret1d).mean().item()
            ic += batch_pearsonr(ensemble_value, self.ret2d).mean().item()
            ic += batch_pearsonr(ensemble_value, self.ret5d).mean().item()
            ic /= 3
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return self.calc_pool_IC_ret(exprs, weights)  # TODO
        with torch.no_grad():
            ensemble_value = self._make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.ret1d).mean().item()
            rank_ic += batch_spearmanr(ensemble_value, self.ret2d).mean().item()
            rank_ic += batch_spearmanr(ensemble_value, self.ret5d).mean().item()
            rank_ic /= 3
            return rank_ic

    ################################# TODO ################################

    def _calc_q90(self, value: Tensor, target: Tensor) -> float:
        top_10_percent_idx = int(0.1 * len(value))
        sorted_indices = torch.argsort(value, descending=True)
        top_10_percent_values = target[sorted_indices[:top_10_percent_idx]]
        return top_10_percent_values.mean().item()

    def calc_q90_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_q90(value, self.ret1d)  # 你可以根据需要选择 ret1d, ret2d 或 ret5d
