import json
import os
from collections import Counter

import numpy as np

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


funcs = [make_function(**func._asdict()) for func in generic_funcs]

instruments = "csi300"
seed = 4
reseed_everything(seed)

cache = {}
device = torch.device("cuda:1")
device_data = torch.device("cpu")
data = ArgData(
    start_time=20190103,
    end_time=20201231,
    device=device_data,
)
data_valid = ArgData(
    start_time=20210101,
    end_time=20210601,
    device=device_data,
)
data_test = ArgData(
    start_time=20210601,
    end_time=20211201,
    device=device_data,
)

pool = AlphaPool(capacity=10, stock_data=data, target=target, ic_lower_bound=None)

target_factor = target.evaluate(data)
target_factor_valid = target.evaluate(data_valid)
target_factor_test = target.evaluate(data_test)


def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count("(") + key.count(")")
    if token_len > 20:
        return -1.0

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor).mean().item()
    except OutOfDataRangeError:
        ic = -1.0
    if np.isnan(ic):
        ic = -1.0
    cache[key] = ic
    return ic


Metric = make_fitness(function=_metric, greater_is_better=True)


def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    v_valid = eval(top_key).evaluate(data_valid)
    v_test = eval(top_key).evaluate(data_test)
    ic_test = batch_pearsonr(v_test, target_factor_test).mean().item()
    ic_valid = batch_pearsonr(v_valid, target_factor_valid).mean().item()
    ric_test = batch_spearmanr(v_test, target_factor_test).mean().item()
    ric_valid = batch_spearmanr(v_valid, target_factor_valid).mean().item()
    return {
        "ic_test": ic_test,
        "ic_valid": ic_valid,
        "ric_test": ric_test,
        "ric_valid": ric_valid,
    }


def try_pool(capacity):
    pool = AlphaPool(
        capacity=capacity, stock_data=data, target=target, ic_lower_bound=None
    )

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {
        "ic_test": ic_test,
        "ic_valid": ic_valid,
        "ric_test": ric_test,
        "ric_valid": ric_valid,
    }


generation = 0


def ev():
    global generation
    generation += 1
    res = [{"pool": 0, "res": try_single()}] + [
        {"pool": cap, "res": try_pool(cap)} for cap in (10, 20, 50, 100)
    ]
    print(res)
    dir_ = "/path/to/save/results"
    os.makedirs(dir_, exist_ok=True)
    if generation % 2 == 0:
        with open(f"{dir_}/{generation}.json", "w") as f:
            json.dump({"cache": cache, "res": res}, f)


if __name__ == "__main__":
    features = ["open_", "close", "high", "low", "volume", "vwap"]
    constants = [
        f"Constant({v})"
        for v in [
            -30.0,
            -10.0,
            -5.0,
            -2.0,
            -1.0,
            -0.5,
            -0.01,
            0.01,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            30.0,
        ]
    ]
    terminals = features + constants

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    est_gp = SymbolicRegressor(
        population_size=1000,
        generations=40,
        init_depth=(2, 6),
        tournament_size=600,
        stopping_criteria=1.0,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=funcs,
        metric=Metric,
        const_range=None,
        n_jobs=1,
    )
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train))
