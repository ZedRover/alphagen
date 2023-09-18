from concurrent.futures import ProcessPoolExecutor
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th

from utils import *


def task_fetch_path(tag, horizon):
    file_names = glob(f"checkpoints/*{tag}*")
    file_names = [i.split("/")[-1] for i in file_names]
    sigs_dir = [
        sorted(
            glob(f"checkpoints/{name}/*.json"),
            key=lambda x: int(x.split("/")[-1].split("_")[0]),
        )[-1]
        for name in file_names
    ]
    return sigs_dir


def task_calc_factors(sigs_dir, horizon=100, num_cores=5):
    with ProcessPoolExecutor(num_cores) as executor:
        resutls = list(
            executor.map(
                json_to_factor,
                sigs_dir,
                [20190103] * len(sigs_dir),
                [20211231] * len(sigs_dir),
                [horizon] * len(sigs_dir),
            )
        )

        new_signal = th.stack([i for i in resutls], dim=-1)

        padding = th.zeros(horizon, 6000, new_signal.shape[-1])
        return th.cat([padding, new_signal], dim=0)


if __name__ == "__main__":
    config_dict = {"tags": ["satd", "ret1d"], "horizon": [100, 320]}
    for i in range(len(config_dict["tags"])):
        sigs_dir = task_fetch_path(config_dict["tags"][i], config_dict["horizon"][i])
        print(sigs_dir)
        sigs = task_calc_factors(sigs_dir, config_dict["horizon"][i])
        np.save(
            f'alphas/{config_dict["tags"][i]}_h{config_dict["horizon"][i]}_alphas.npy',
            sigs.numpy().astype(np.float32),
        )
        np.save(
            f'alphas/{config_dict["tags"][i]}_h{config_dict["horizon"][i]}_dirs.npy',
            sigs_dir,
        )
