import json
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import List, Union

import pandas as pd
import ray
import torch as th
from audtorch.metrics.functional import pearsonr

from alphagen.config import OPERATORS
from alphagen.data.expression import *
from alphagen.data.expression_ocean import *
from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils.correlation import batch_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.calculator import QLibStockDataCalculator
from alphagen_ocean.stock_data import ArgData, FeatureType


def tokenize_formula(formula: str) -> List[str]:
    tokens = []
    token = ""
    i = 0
    while i < len(formula):
        char = formula[i]
        if char == "(" or char == "," or char == ")":
            if token:
                tokens.append(token.strip())
                token = ""
            tokens.append(char)
            i += 1
        elif char == "$":
            token += char
            i += 1
            while i < len(formula) and formula[i].isalnum():
                token += formula[i]
                i += 1
        elif formula[i : i + 8] == "Constant":
            i += 8
            const_val = ""
            while formula[i] != ")":
                const_val += formula[i]
                i += 1
            token = "Constant" + const_val
            tokens.append(token.strip())
            token = ""
            i += 1  # skip the closing parenthesis
        else:
            token += char
            i += 1
    if token:
        tokens.append(token.strip())
    return tokens


def infix_to_rpn(
    tokens: List[str], operators: List[Type[Operator]] = OPERATORS
) -> List[str]:
    output = []
    stack = []

    def is_operator(token: str) -> bool:
        return any(op.__name__ == token for op in operators)

    for token in tokens:
        if (
            token.startswith("$")
            or token.isdigit()
            or (token[0] == "-" and token[1:].isdigit())
            or token.startswith("Constant")
        ):
            output.append(token)
        elif is_operator(token):
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ",":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()  # remove the open parenthesis
        else:
            raise ValueError(f"Unknown token: {token}")

    while stack:
        output.append(stack.pop())
    return output


def formula_to_expression(formula: str) -> Expression:
    tokens = tokenize_formula(formula)
    rpn_tokens = infix_to_rpn(tokens)
    builder = ExpressionBuilder()
    for token in rpn_tokens:
        if token.startswith("$"):
            feature_type = getattr(FeatureType, token[1:])
            builder.add_token(FeatureToken(feature_type))
        elif token.startswith("Constant"):
            value = float(token[9:])
            builder.add_token(ConstantToken(value))
        elif token.isdigit() or (token[0] == "-" and token[1:].isdigit()):
            try:
                builder.add_token(DeltaTimeToken(int(token)))
            except:
                builder.add_token(ConstantToken(int(token)))
        else:
            operator_class = next(
                (op for op in OPERATORS if op.__name__ == token), None
            )
            if operator_class:
                builder.add_token(OperatorToken(operator_class))
            else:
                raise ValueError(f"Unknown operator: {token}")

    if not builder.is_valid():
        raise ValueError("Invalid formula")

    return builder.get_tree()


#########################################################################################


def calc_topk(input: Tensor, target: Tensor, topk: int = 10) -> Tensor:
    idx = ~torch.isnan(target)
    input, target = input[idx], target[idx]
    top_q_indices = torch.topk(
        input, int(len(target) * topk / 100), largest=True
    ).indices
    q_ret = torch.nanmean(target[top_q_indices])
    return q_ret


def batch_topk(yhat, y):
    q90 = []
    q99 = []
    for i in range(len(yhat)):
        yhat_ = yhat[i]
        y_ = y[i]
        q90.append(calc_topk(yhat_, y_, 10).item())
        q99.append(calc_topk(yhat_, y_, 1).item())
    q99 = np.mean(np.array(q99))
    q90 = np.mean(np.array(q90))
    return round(q90, 5), round(q99, 5)


def json_to_factor(path):
    data = ArgData(
        start_time=20190103,
        end_time=20211231,
        device=torch.device("cpu"),
        max_backtrack_days=HORIZON_BACK,
    )
    with open(path, "r") as f:
        alpha = json.load(f)
    factors = [
        Feature(getattr(FeatureType, expr[1:])).evaluate(data)
        if expr[0] == "$"
        else formula_to_expression(expr).evaluate(data)
        for expr in alpha["exprs"]
    ]
    weights = torch.tensor(alpha["weights"])
    factor_value = sum(f * w for f, w in zip(factors, weights))
    factor_value = normalize_by_day(factor_value)
    return factor_value


def calc_factors(data: ArgData, path: str):
    with open(path, "r") as f:
        alpha = json.load(f)
    factors = [
        Feature(getattr(FeatureType, expr[1:])).evaluate(data)
        if expr[0] == "$"
        else formula_to_expression(expr).evaluate(data)
        for expr in alpha["exprs"]
    ]
    weights = torch.tensor(alpha["weights"])
    factor_value = sum(f * w for f, w in zip(factors, weights))
    factor_value = normalize_by_day(factor_value)
    return factor_value


@ray.remote
def remote_json_to_factor(path, start_time, end_time):
    return json_to_factor(path, start_date=start_time, end_date=end_time)


@ray.remote
def remote_batch_pearsonr(tensor1, tensor2):
    return th.nanmean(batch_pearsonr(tensor1, tensor2)).item()


@ray.remote
def remote_calc_factors(data, path):
    return calc_factors(data, path)


@ray.remote
def aud_pearsonr(tensor1, tensor2) -> float:
    return th.nanmean(pearsonr(tensor1, tensor2)).item()


def remove_low_ic_factors(df_ic_insample, df_ic_outsample, strict=True):
    if strict:
        print("2019-2020")
        df_ic_insample = df_ic_insample.reset_index(drop=True)
        df_ic_outsample = df_ic_outsample.reset_index(drop=True)
        decay = df_ic_insample - df_ic_outsample
        idx = (decay / df_ic_insample < 0.5).all(axis=1) & (df_ic_insample > 0).all(
            axis=1
        )
        print(df_ic_insample[idx])
        print("2021")
        print(df_ic_outsample[idx])
    return df_ic_outsample[idx].index.tolist()


def remove_high_corr_factors(df_corr, threshold=0.8):
    factors = list(range(df_corr.shape[0]))
    high_corr_pairs = np.where(np.triu(df_corr, 1) > threshold)
    high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs)]

    removed_factors = set()

    while high_corr_pairs:
        avg_corrs = {}
        for factor1, factor2 in high_corr_pairs:
            avg_corr_factor1 = np.mean(
                [df_corr[factor1, k] for k in factors if k not in [factor1, factor2]]
            )
            avg_corr_factor2 = np.mean(
                [df_corr[factor2, k] for k in factors if k not in [factor1, factor2]]
            )
            avg_corrs[(factor1, factor2)] = (avg_corr_factor1, avg_corr_factor2)

        for (factor1, factor2), (
            avg_corr_factor1,
            avg_corr_factor2,
        ) in avg_corrs.items():
            if avg_corr_factor1 > avg_corr_factor2:
                removed_factors.add(factor1)
            else:
                removed_factors.add(factor2)

        factors = [f for f in factors if f not in removed_factors]

        high_corr_pairs = np.where(np.triu(df_corr[factors][:, factors], 1) > threshold)
        high_corr_pairs = [(factors[i], factors[j]) for i, j in zip(*high_corr_pairs)]

    return factors


# timer wrapper
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {round(end-start)} seconds")
        return result

    return wrapper


class Backtester(object):
    def __init__(
        self,
        start_time: int = 20190103,
        end_time: int = 20190605,
        pattern: str = "./checkpoints/*sat*/",
        **data_param,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.paths = sorted(glob(pattern))
        self.json_paths = [
            sorted(
                glob(path + "*.json"), key=lambda x: int(x.split("/")[-1].split("_")[0])
            )[-1]
            for path in self.paths
        ]
        print(f"num of factors:{len(self.json_paths)}")
        self.data_test = ArgData(self.start_time, self.end_time)
        self.calter = QLibStockDataCalculator(self.data_test, **data_param)
        self.factor_names = [path.split("/")[-2] for path in self.json_paths]

    @timer
    def calc_factor(self):
        n = len(self.json_paths)
        with ProcessPoolExecutor(max_workers=160) as executor:
            factors = executor.map(
                json_to_factor,
                self.json_paths,
                [self.start_time] * n,
                [self.end_time] * n,
            )
            factors = list(factors)
        self.factors = factors
        return factors

    @timer
    def calc_corr(self):
        sigs = self.factors
        n = len(sigs)
        df_corr = np.ones((n, n))
        corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                ic = torch.nanmean(
                    batch_pearsonr(
                        sigs[i],
                        sigs[j],
                    )
                )
                df_corr[i, j] = ic
                df_corr[j, i] = ic
        self.df_corr = df_corr
        return df_corr

    @timer
    def calc_ic(self):
        y1 = self.calter.raw_ret1d
        y2 = self.calter.raw_ret2d
        y5 = self.calter.raw_ret5d
        futures = []
        for y in self.factors:
            futures.append(remote_batch_pearsonr.remote(y, y1))
            futures.append(remote_batch_pearsonr.remote(y, y2))
            futures.append(remote_batch_pearsonr.remote(y, y5))
        results = ray.get(futures)
        data = {"y1": results[::3], "y2": results[1::3], "y5": results[2::3]}
        df_ic = pd.DataFrame(data)
        index_name = pd.Index([path.split("/")[-2] for path in self.json_paths])
        df_ic.index = index_name
        self.df_ic = df_ic

    def make(self):
        self.calc_factor()
        self.calc_corr()
        self.calc_ic()
        return self.factors, self.df_corr, self.df_ic


class DoubleChecker(object):
    def __init__(self, raw: Backtester, new: Backtester, strict=True):
        self.raw = raw
        self.new = new
        self.strict = strict
        self.factors_idx = list(range(len(new.factors)))

    def filter_ic(
        self,
    ):
        return remove_low_ic_factors(self.raw.df_ic, self.new.df_ic, self.strict)

    def filter_corr(
        self,
    ):
        raw_factors = remove_high_corr_factors(self.raw.df_corr)
        new_factors = remove_high_corr_factors(self.new.df_corr)
        return list(set(raw_factors) & set(new_factors))

    def make(self):
        ic_idx = self.filter_corr()
        corr_idx = self.filter_ic()
        result_idx = list(set(ic_idx) & set(corr_idx))
        return result_idx


if __name__ == "__main__":
    formula = "Add($qnet_incr_cash_cash_equ_ttm,Greater(DeStd(Deltaratio(Constant(-0.5),$qbuy_value_exlarge_order_act)),BiasCrsRank($qnet_inflow_rate_value)))"
    expression = formula_to_expression(formula)
    print(expression)
    data_test = ArgData(
        start_time=20210101,
        end_time=20211231,
        device=th.device("cpu"),
    )
    s = time.time()
    expression.evaluate(data_test)
    e = time.time()
    print("Signal calculation time:", e - s)
