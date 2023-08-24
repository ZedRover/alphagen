import torch
from torch import Tensor

from alphagen.utils.pytorch_utils import masked_mean_std
import torchsort


def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()  # [days, stocks]
    y = y.clone()  # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    rank = x.argsort().argsort().float()  # [d, s]
    eq = x[:, None] == x[:, :, None]  # [d, s, s]
    eq = eq / eq.sum(dim=2, keepdim=True)  # [d, s, s]
    rank = (eq @ rank[:, :, None]).squeeze(dim=2)
    rank[nan_mask] = 0
    return rank  # [d, s]


def _batch_pearsonr_given_mask(x: Tensor, y: Tensor, n: Tensor, mask: Tensor) -> Tensor:
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    return corrs


# write a timer func wrapper , print the clean name of func and time, and check the env of the func, if it's in jupyter notebook then print, if it's called by a main func then not print
def timer(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} cost {end - start} seconds")
        return ret

    return wrapper


@timer
def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


# @timer
def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.0))


import time
from dataclasses import dataclass
from functools import wraps
from typing import List

import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
from numba import njit, prange


def timeit():
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            interval = time.time() - start_time
            print("Time for '%s': %0.3f seconds" % (func.__qualname__, interval))
            return res

        return inner

    return outer


@njit(fastmath=True, parallel=True)
def mean1(a):
    n = len(a)
    b = np.empty(n)
    for i in prange(n):
        b[i] = a[i].mean()
    return b


@njit(fastmath=True, parallel=True)
def std1(a):
    n = len(a)
    b = np.empty(n)
    for i in prange(n):
        b[i] = a[i].std()
    return b


@njit(fastmath=False, parallel=True)
def numba_corrcoef(a):
    """Correlation"""
    n, k = a.shape

    mu_a = mean1(a)
    sig_a = std1(a)

    for i in prange(n):
        a[i] = (a[i] - mu_a[i]) / sig_a[i]

    ak = a / k

    out = np.empty((n, n))

    for i in prange(n):
        out[i, i] = 1.0
        for j in prange(i + 1, n):
            out[i, j] = ak[i] @ a[j]
            out[j, i] = out[i, j]

    return out


@timeit()
def pd_corr_numba(
    feature_matrix: pd.DataFrame, dtype=np.float64, copy: bool = False
) -> pd.DataFrame:
    return pd.DataFrame(
        numba_corrcoef(feature_matrix.to_numpy(dtype=dtype, copy=copy).T),
        index=feature_matrix.columns,
        columns=feature_matrix.columns,
    )


@timeit()
def drop_correlated_numeric_columns(
    feature_matrix: pd.DataFrame, thresh: float
) -> pd.DataFrame:
    if thresh is None or thresh == 0:
        return feature_matrix
    corr = pd_corr_numba(feature_matrix=feature_matrix, dtype=np.float32)
    return drop_correlated_columns(corr, feature_matrix, thresh)


@timeit()
def drop_correlated_categorical_columns(
    feature_matrix: pd.DataFrame, thresh: float
) -> pd.DataFrame:
    if thresh is None or thresh == 0:
        return feature_matrix
    corr = pd_cramers(df=feature_matrix)
    return drop_correlated_columns(corr, feature_matrix, thresh)


def drop_correlated_columns(corr, df: pd.DataFrame, thresh) -> pd.DataFrame:
    prefix = f"In {drop_correlated_columns.__qualname__} --"
    to_drop = get_correlated_columns_to_drop(corr=corr, thresh=thresh)

    print(f"{prefix} Before: feats_n_labs.shape={df.shape}")
    df = df.drop(columns=to_drop)
    print(f"{prefix} After: df.shape={df.shape}")
    return df


@dataclass
class CorrFilterResult:
    df: pd.DataFrame
    corr: pd.DataFrame
    thresh: float
    dropped: List[str]


def print_features_most_correlated_with_label(corr: pd.DataFrame) -> None:
    prefix = f"In {print_features_most_correlated_with_label.__qualname__} --"
    label_correlations: pd.Series = corr["label"].sort_values()
    label_correlations: pd.Series = label_correlations[~pd.isna(label_correlations)]
    print(f"{prefix} label_correlations.to_dict()={label_correlations.to_dict()}")


@timeit()
def get_correlated_columns_to_drop(corr: pd.DataFrame, thresh: float) -> List[str]:
    corr = corr.abs()

    # Subset to the upper triangle of correlation matrix
    upper_tri_idxs = np.triu(np.ones(corr.shape), k=1).astype(np.bool)
    corr_upper_tri = corr.where(upper_tri_idxs)

    # Identify names of columns with correlation above threshold
    return [
        column
        for column in corr_upper_tri.columns
        if (corr_upper_tri[column] >= thresh).any() and column != "label"
    ]


def pd_cramers(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            print(col)
            raise ValueError(col)
    dtype_is_cat = df.dtypes == "category"
    print(dtype_is_cat)
    # noinspection PyUnresolvedReferences
    assert len(dtype_is_cat.drop_duplicates()) == 1
    # noinspection PyUnresolvedReferences
    assert dtype_is_cat.drop_duplicates().iloc[0] == True

    ncols = df.shape[1]
    nsamples = df.shape[0]

    df_np = np.empty((ncols, nsamples), dtype=np.int32)
    ncats = np.empty(ncols, dtype=np.int32)
    for i in range(ncols):
        col = df[df.columns[i]]

        codes = col.cat.codes.to_numpy(np.int32)
        if -1 in set(codes):
            codes += 1  # So they're all indices
        assert set(codes) == set(range(np.max(codes) + 1)), set(codes) - set(
            range(np.max(codes) + 1)
        )

        df_np[i] = codes
        ncats[i] = np.unique(codes).size

    out = cramer_many(ncats, df_np)
    return pd.DataFrame(data=out, index=df.columns, columns=df.columns)


@njit(parallel=True)
def cramer_many(ncats, df_np):
    ncols = ncats.size
    out = np.empty((ncols, ncols))
    for i in prange(ncols):
        out[i, i] = 1.0
        col_i = df_np[i]
        ncats_i = ncats[i]
        for j in prange(i + 1, ncols):
            col_j = df_np[j]
            ncats_j = ncats[j]

            out_ij = cramers_corrected_stat(ncats_i, ncats_j, col_i, col_j)

            out[i, j] = out_ij
            out[j, i] = out[i, j]

    return out


@njit(parallel=True)
def nb_crosstab(ncats1, ncats2, col1, col2):
    nsamples = col1.size
    out = np.zeros((ncats1, ncats2), dtype=np.float32)
    for i in prange(nsamples):
        col1_cat = col1[i]
        col2_cat = col2[i]
        out[col1_cat, col2_cat] += 1.0
    return out


@njit(parallel=True)
def calculate_chi2(ncats1, ncats2, f_obs):
    sums1 = np.sum(f_obs, axis=1)  # sum over each row. Has size=ncats1
    sums2 = np.sum(f_obs, axis=0)  # sum over each column. Has size=ncats2
    total = np.sum(f_obs)
    out = 0
    for i in prange(ncats1):
        p_i = sums1[i] / total
        for j in prange(ncats2):
            f_expected = p_i * sums2[j]
            surprise = f_obs[i, j] - f_expected
            out += (surprise**2) / f_expected
    return out


@njit(parallel=False)
def cramers_corrected_stat(ncats1, ncats2, col1, col2):
    cf = nb_crosstab(ncats1, ncats2, col1, col2)
    chi2 = calculate_chi2(ncats1, ncats2, cf)
    return _cramers_corrected_stat(chi2, cf)


@njit(parallel=True)
def _cramers_corrected_stat(chi2, confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
