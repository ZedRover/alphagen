import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from alphagen_ocean.calculator_t0 import Calculator_t0
import numpy as np
import pandas as pd
import torch as th
import os
from utils import *


def json_to_factor(path, start_time, end_time, max_backtrack_days):
    try:
        calculator = Calculator_t0(
            codes=SELECTED_CODES,
            start=start_time,
            end=end_time,
            max_backtrack_ticks=max_backtrack_days,
            max_future_ticks=0,
            device=torch.device("cpu"),
        )
        data_dict = calculator.dat_dict
        with open(path, "r") as f:
            alpha = json.load(f)
        # TODO
        ret = []
        for code in data_dict:
            data = data_dict[code]
            factors = [
                (
                    Feature(getattr(FeatureType, expr[1:])).evaluate(data)
                    if expr[0] == "$"
                    else formula_to_expression(expr).evaluate(data)
                )
                for expr in alpha["exprs"]
            ]
            weights = torch.tensor(alpha["weights"])
            factor_value = sum(f * w for f, w in zip(factors, weights))
            padding = th.zeros(max_backtrack_days, 1)
            ret.append(th.concat([padding, factor_value], dim=0))
    except Exception as e:
        # print(path.split("/")[-2] + " error" + e)
        print(f"Error,{path}")
    print(path)

    return ret


if __name__ == "__main__":
    start_time = 20210101
    end_time = 20221231
    max_backtrack_days = 200

    sigs_dir = glob("release_json/*.json")
    sigs_dir = sorted(sigs_dir, key=os.path.getmtime)

    with ProcessPoolExecutor(20) as executor:
        results = list(
            executor.map(
                json_to_factor,
                sigs_dir,
                [start_time] * len(sigs_dir),
                [end_time] * len(sigs_dir),
                [max_backtrack_days] * len(sigs_dir),
            )
        )

    signals_dict = {}
    for i in range(len(SELECTED_CODES)):
        code = SELECTED_CODES[i]
        signals_dict[code] = th.stack([j[i].reshape(-1, 1) for j in results], dim=1)

    for code in SELECTED_CODES:
        sig_now = np.squeeze(signals_dict[code])
        np.savetxt(
            f"t0/stkCode_{code}.csv",
            sig_now.numpy().astype(np.float32),
            delimiter=",",
            comments="",
            fmt="%f",
        )
