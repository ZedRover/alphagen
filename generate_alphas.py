from concurrent.futures import ProcessPoolExecutor
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th

from utils import *

df_result = pd.read_csv("./result.csv", index_col=0)
sigs_name = df_result.index.tolist()
sigs_dir = [
    sorted(
        glob(f"checkpoints/{name}/*.json"),
        key=lambda x: int(x.split("/")[-1].split("_")[0]),
    )[-1]
    for name in sigs_name
]
sigs_dict = {}
with ProcessPoolExecutor(80) as executor:
    for name, value in zip(
        sigs_name,
        executor.map(
            json_to_factor,
            sigs_dir,
            [20190103] * len(sigs_name),
            [20211231] * len(sigs_name),
        ),
    ):
        sigs_dict[name] = value
