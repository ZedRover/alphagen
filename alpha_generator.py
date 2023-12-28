import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import pandas as pd
import torch as th

from utils import *


def json_to_factor(path, start_time, end_time, max_backtrack_days):
    try:
        data = ArgData(
            start_time=start_time,
            end_time=end_time,
            max_backtrack_days=max_backtrack_days,
            device=torch.device("cpu"),
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
        padding = th.zeros(max_backtrack_days, 6000)
        print(path.split("/")[-2] + " done")
        return th.concat([padding, factor_value], dim=0)
    except Exception as e:
        print(path.split("/")[-2] + " error" + e)
        return None


def task_fetch_path(tag, ckpt_path="checkpoints"):
    file_names = glob(f"{ckpt_path}/*{tag}*")
    file_names = [i.split("/")[-1] for i in file_names]
    sigs_dir = [
        sorted(
            glob(f"{ckpt_path}/{name}/*.json"),
            key=lambda x: int(x.split("/")[-1].split("_")[0]),
        )[-1]
        for name in file_names
    ]
    return sigs_dir  # test


def task_calc_factors(
    sigs_dir, horizon=100, num_cores=5, start_time=20190103, end_time=20211231
):
    with ProcessPoolExecutor(num_cores) as executor:
        resutls = list(
            executor.map(
                json_to_factor,
                sigs_dir,
                [start_time] * len(sigs_dir),
                [end_time] * len(sigs_dir),
                [horizon] * len(sigs_dir),
            )
        )

        new_signal = th.stack([i for i in resutls], dim=-1)
        return new_signal


if __name__ == "__main__":
    # config_dict = {"tags": ["satd", "ret1d"], "horizon": [100, 306]}
    config_dict = {"tags": ["ret1d"], "horizon": [130]}
    num_cores = 15
    start_time = 20190103
    end_time = 20211231

    for i in range(len(config_dict["tags"])):
        tag = config_dict["tags"][i]
        horizon = config_dict["horizon"][i]
        s = time.time()
        sigs_dir = task_fetch_path(tag, ckpt_path="checkpoints_10d")
        with open(f"alphas/{tag}_h{horizon}_alphas.txt", "w") as f:
            f.write(f"{sigs_dir}")
        print(f"tag:{tag} horizon:{horizon} len_sigs_dir:{len(sigs_dir)}")
        sigs = task_calc_factors(
            sigs_dir, horizon, num_cores, start_time=start_time, end_time=end_time
        )
        np.save(
            f"alphas/{tag}_h{horizon}_alphas.npy",
            sigs.numpy().astype(np.float32),
        )
        with open(f"alphas/{tag}_h{horizon}_alphas.txt", "a") as f:
            f.write(f"{sigs_dir}")
        e = time.time()
        print(
            f"num_alphas:{len(sigs_dir)},tag:{tag},horizon:{horizon},time:{e-s}s,num_cores:{num_cores}"
        )
        print("-" * 50)
