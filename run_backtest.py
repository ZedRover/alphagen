# %%
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import torch as th
import wandb

from utils import *

# %%
ray.init(num_cpus=160, ignore_reinit_error=True)
wandb.init(project="alpha_backtest_daily")
# %%
insmp = Backtester(
    20190103,
    20210101,
    pattern="./checkpoints/*sat*/",
)
otsmp = Backtester(
    20210101,
    20211231,
    pattern="./checkpoints/*sat*/",
)

# %%
factors = insmp.calc_factor()
factors2 = otsmp.calc_factor()
wandb.log({"total_sigs_num": len(factors)})
# %%
df_ic = insmp.calc_ic()
df_ic2 = otsmp.calc_ic()


# %%
ic_filter_idx = otsmp.df_ic.query(
    "y1>0.01 and y2>0.01 and y5>0.01 and @insmp.df_ic.y1>.01 and @insmp.df_ic.y2>.015 and @insmp.df_ic.y5>.02"
).index

wandb.log({"high_ic_sigs": len(ic_filter_idx)})
wandb.log({"len_ic_sigs": len(otsmp.factor_names)})
# %%
feature_dict = dict(zip(otsmp.factor_names, otsmp.factors))

# %%
ic_filter_sigs = [feature_dict[i] for i in ic_filter_idx]


def calc_single_corr(i, j, sigs):
    return torch.nanmean(batch_pearsonr(sigs[i], sigs[j]))


def calc_sig_corr(sigs):
    n = len(sigs)
    df_corr = np.ones((n, n))

    with ProcessPoolExecutor() as executor:
        future_to_index = {
            (i, j): executor.submit(calc_single_corr, i, j, sigs)
            for i in range(n)
            for j in range(i + 1, n)
        }

        for (i, j), future in future_to_index.items():
            ic = future.result()
            df_corr[i, j] = ic
            df_corr[j, i] = ic

    return df_corr


# %%
df_corr = calc_sig_corr(ic_filter_sigs)

# %%
df_corr = pd.DataFrame(df_corr)

# %%
lowcorr_idx = remove_high_corr_factors(df_corr.values)
low_cor_sigs = [ic_filter_sigs[idx] for idx in lowcorr_idx]
wandb.log({"low_corr_sigs": len(lowcorr_idx)})
wandb.log({"len_corr_sigs": len(ic_filter_idx)})

# %%
final = [ic_filter_idx[idx] for idx in lowcorr_idx]

# %%

# 设置显示最大列数
pd.set_option("display.max_columns", None)

# 设置显示最大行数
pd.set_option("display.max_rows", None)
finaldf = otsmp.df_ic.loc[final]

# %%
finaldf.describe()

# %%
fs = ic_filter_sigs
y1d = otsmp.calter.raw_ret1d
y2d = otsmp.calter.raw_ret2d
y5d = otsmp.calter.raw_ret5d

# %%
finaldf

# %%
finaldf.to_csv("result.csv")

# %%
df_corr.iloc[lowcorr_idx, lowcorr_idx].to_csv("df_corr.csv")

# %%
df_corr.replace(1, 0).describe()

# %%
len(finaldf)

# %%


# %%
