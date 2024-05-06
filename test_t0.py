import time
import json
import torch as th
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from alphagen_ocean.calculator_t0 import Calculator_t0
from alphagen.utils.pytorch_utils import normalize_timeseries
from utils import *

last = True
config_dict = {"tags": ["ret1d"], "horizon": [4000]}
num_cores = 12
use_multiprocessing = True
start_time = 20210101
end_time = 20221231
save_path = "last_signal"


def json_to_factor(path, start_time, end_time, max_backtrack_days):
    try:
        calculator = Calculator_t0(
            codes=SELECTED_CODES,
            start=start_time,
            end=end_time,
            max_backtrack_ticks=max_backtrack_days,
            max_future_ticks=0,
            device=th.device("cpu"),
        )
        data_dict = calculator.dat_dict
        with open(path, "r") as f:
            alpha = json.load(f)
        ret = {}
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
            weights = th.tensor(alpha["weights"])
            factor_value = sum(
                normalize_timeseries(f) * w for f, w in zip(factors, weights)
            )
            factor_value = normalize_timeseries(factor_value)
            padding = th.zeros(max_backtrack_days, 1)
            ret[code] = th.concat([padding, factor_value], dim=0)
        print(f"{path.split('/')[-2]} done")
    except Exception as e:
        print(f"{path.split('/')[-2]} error: {str(e)}")
    return ret


def task_fetch_path(tag, ckpt_path="checkpoints"):
    file_names = glob(f"{ckpt_path}/*{tag}*")
    file_names = [i.split("/")[-1] for i in file_names]
    sigs_dir = []
    for name in file_names:
        if last == False:
            sigs_dir.extend(
                sorted(
                    glob(f"{ckpt_path}/{name}/*.json"),
                    key=lambda x: int(x.split("/")[-1].split("_")[0]),
                )
            )
        else:
            sigs_dir.append(
                sorted(
                    glob(f"{ckpt_path}/{name}/*.json"),
                    key=lambda x: int(x.split("/")[-1].split("_")[0]),
                )[-1]
            )

    return sigs_dir


def task_calc_factors(
    sigs_dir,
    horizon=100,
    use_multiprocessing=True,
    num_cores=5,
    start_time=20190103,
    end_time=20211231,
):
    if use_multiprocessing:
        with ProcessPoolExecutor(num_cores) as executor:
            results = list(
                executor.map(
                    json_to_factor,
                    sigs_dir,
                    [start_time] * len(sigs_dir),
                    [end_time] * len(sigs_dir),
                    [horizon] * len(sigs_dir),
                )
            )
    else:
        results = [
            json_to_factor(sig, start_time, end_time, horizon) for sig in sigs_dir
        ]
    signals_dict = {}
    for i in range(len(SELECTED_CODES)):
        code = SELECTED_CODES[i]
        signals_dict[code] = th.stack([j[code].reshape(-1, 1) for j in results], dim=1)
    return signals_dict


if __name__ == "__main__":

    for i in range(len(config_dict["tags"])):
        tag = config_dict["tags"][i]
        horizon = config_dict["horizon"][i]
        s = time.time()
        sigs_dir = task_fetch_path(tag, ckpt_path="t0_results")[:2]
        print(f"tag:{tag} horizon:{horizon} len_sigs_dir:{len(sigs_dir)}")
        sigs = task_calc_factors(
            sigs_dir,
            horizon,
            use_multiprocessing,
            num_cores,
            start_time=start_time,
            end_time=end_time,
        )
        for code in SELECTED_CODES:
            sig_now = sigs[code]
            print(sig_now.shape)
            np.savetxt(
                f"{save_path}/stkCode_{code}.csv",
                sig_now.squeeze().numpy().astype(np.float32),
                delimiter=",",
                comments="",
                fmt="%f",
            )
        e = time.time()
        print(
            f"num_alphas:{len(sigs_dir)},tag:{tag},horizon:{horizon},time:{e-s}s,num_cores:{num_cores}"
        )
        print("-" * 50)
