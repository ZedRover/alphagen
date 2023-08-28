import numpy as np
import torch as th
from concurrent.futures import ProcessPoolExecutor

from glob import glob
from utils import *
import ray
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_result = pd.read_csv("./result.csv", index_col=0)
sigs_name = df_result.index.tolist()
sigs_dir = [
    sorted(
        glob(f"checkpoints/{name}/*.json"),
        key=lambda x: int(x.split("/")[-1].split("_")[0]),
    )[-1]
    for name in sigs_name
]


with ProcessPoolExecutor(160) as executor:
    sigs_value = executor.map(
        json_to_factor,
        sigs_dir,
        [20210101] * len(sigs_name),
        [20211231] * len(sigs_name),
    )
    sigs_value = list(sigs_value)
