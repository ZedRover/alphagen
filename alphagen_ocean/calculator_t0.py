from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch import Tensor

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.dir_config import DIR_RETS
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, pool_pearsonr
from alphagen.utils.pytorch_utils import *
from alphagen_ocean.stock_data import *
import SharedArray as sa
from time import time


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def filter_data(
    label, timestamp, start, end, max_backtrack_ticks, max_future_ticks, device
):
    start_idx, end_idx = find_date_indices(timestamp, start, end)
    real_start_idx = start_idx + max_backtrack_ticks
    real_end_idx = end_idx - max_future_ticks
    ret = label[real_start_idx:real_end_idx].reshape(-1, 1)
    return torch.from_numpy(ret).to(device).nan_to_num(0, 0, 0)


class Calculator_t0(AlphaCalculator):
    def __init__(
        self,
        codes=["000537"],
        start=20210101,
        end=20211231,
        max_backtrack_ticks=128,
        max_future_ticks=0,
        device=torch.device("cuda:0"),
    ):
        self.device = device
        self.codes = codes

        s = time()
        self.dat_dict = {
            code: FakeData(
                code=code,
                start_time=start,
                end_time=end,
                max_backtrack_days=max_backtrack_ticks,
                max_future_days=max_future_ticks,
                device=self.device,
            )
            for code in codes
        }
        e = time()
        print(f"Load data cost {int(e-s)} s")
        self.ret_dict = {code: self.dat_dict[code].label for code in codes}

    # @timer_func
    @lru_cache()
    def _calc_alpha(self, expr: Expression) -> dict:
        alpha_dict = {}
        for code in self.codes:
            self.dat_dict[code]._data = self.dat_dict[code]._data.to(self.device)
            # 评估表达式并立即将结果移动到 CPU
            with torch.no_grad():
                evaluated_data = expr.evaluate(self.dat_dict[code])
                standardized_data = standardize_tensor(evaluated_data).nan_to_num(
                    0, 0, 0
                )
            self.dat_dict[code]._data = self.dat_dict[code]._data.to("cpu")
            alpha_dict[code] = standardized_data.to("cpu")
            del evaluated_data
            torch.cuda.empty_cache()
        return alpha_dict

    # @timer_func
    @lru_cache(maxsize=50)
    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        ic = batch_pearsonr(value1.T, value2.T).item()
        if np.isnan(ic):
            print("Error: nan IC".center(30, "="))
            # print(f"value 1:{value1[-10:]}")
            # print(f"value 2:{value2[-10:]}")
        return ic

    # @timer_func
    @lru_cache(maxsize=50)
    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1.T, value2.T).item()

    # @timer_func
    def _make_ensemble_alpha(
        self, exprs: List[Expression], weights: List[float]
    ) -> Tensor:
        n = len(exprs)
        alpha_dicts = []
        for i in range(n):
            alpha_dicts.append(self._calc_alpha(exprs[i]))

        for i in range(n):
            for code in self.codes:
                alpha_dict = alpha_dicts[i]
                alpha_dict[code] *= weights[i]

        factor_dict = {}
        for code in self.codes:
            factor_dict[code] = standardize_tensor(
                sum([alpha_dicts[i][code] for i in range(n)])
            )
        return factor_dict

    # @timer_func
    @lru_cache(maxsize=50)
    def calc_single_IC_ret(self, expr: Expression) -> float:
        alpha_dict = self._calc_alpha(expr)
        ic_dict = {}
        for code in self.codes:
            ic = self._calc_IC(alpha_dict[code], self.ret_dict[code])
            # print(f"calc_single_IC_ret:{ic}")
            ic_dict[code] = ic
        mean_ic = np.nanmean(list(ic_dict.values()))

        return mean_ic

    # @timer_func
    @lru_cache(maxsize=50)
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:

        alpha_dict1, alpha_dict2 = self._calc_alpha(expr1), self._calc_alpha(expr2)

        ics = []
        for code in self.codes:
            ic = self._calc_IC(alpha_dict1[code], alpha_dict2[code])
            if np.isnan(ic):
                print("cal_mutual_IC NaN")
            ics.append(ic)
        return np.nanmean(ics)

    # @timer_func
    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            factors = self._make_ensemble_alpha(exprs, weights)
            ic = 0
            for code in self.codes:
                ic += (
                    batch_pearsonr(factors[code].T, self.ret_dict[code].T).mean().item()
                )
            return ic / len(factors)

    # @timer_func
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        "First combine the alphas linearly,"
        "then Calculate Rank IC between the linear combination and a predefined target."
        return None

    # @timer_func
    def calc_pool_pIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return None
