import argparse
import json

import numpy as np
import torch
from rich.pretty import pprint

from alphagen.config import *
from alphagen.data.expression import *
from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils.correlation import batch_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.calculator import QLibStockDataCalculator
from alphagen_ocean.stock_data import ArgData
from utils import formula_to_expression

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    "-d",
    type=str,
    default="checkpoints/20230811142817_ocean_lexpr8_lopt34_10_513/26624_steps_pool.json",
)
parser.add_argument("--gpu", "-g", type=int, default=3)
args = parser.parse_args()


device = torch.device("cpu")
cuda = torch.device(f"cuda:{args.gpu}")
data_test = ArgData(
    start_time=20210101,
    end_time=20211231,
    device=device,
)
cal_test = QLibStockDataCalculator(data_test)

with open(args.dir, "r") as f:
    alpha = json.load(f)

n_pool = len(alpha["weights"])
factors = [0] * n_pool
for i in range(n_pool):
    if alpha["exprs"][i][0] == "$":
        factors[i] = Feature(getattr(FeatureType, alpha["exprs"][i][1:])).evaluate(
            data_test
        )
    else:
        factors[i] = formula_to_expression(alpha["exprs"][i]).evaluate(data_test)

weights = torch.Tensor(alpha["weights"])
factor_value = torch.zeros_like(factors[0])
for i in range(n_pool):
    factor_value += factors[i] * weights[i]


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
    q90 = np.mean(q90)
    q99 = np.mean(q99)
    return round(q90, 5), round(q99, 5)


yhat = normalize_by_day(factor_value).to(cuda)
y1d = cal_test.raw_ret1d.to(cuda)
y2d = cal_test.raw_ret2d.to(cuda)
y5d = cal_test.raw_ret5d.to(cuda)

metric = {}
rets = ["ret1d", "ret2d", "ret5d"]
ys = [y1d, y2d, y5d]
for i in range(3):
    y = ys[i]
    metric[rets[i]] = {}
    metric[rets[i]]["IC"] = round(torch.nanmean(batch_pearsonr(yhat, y)).item(), 5)
    metric[rets[i]]["Q90"], metric[rets[i]]["Q99"] = batch_topk(yhat, y)


pprint(metric)
