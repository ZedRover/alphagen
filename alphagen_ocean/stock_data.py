from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# import pandas_market_calendars as mcal

import SharedArray as sa
import torch
from sklearn.model_selection import train_test_split
import h5py
from alphagen.dir_config import DIR_DATES

from .feature_list import FEATURES

N_PROD = 6000
MULTI_TI = 16

FeatureType = Enum("FeatureType", {feature: i for i, feature in enumerate(FEATURES)})

# def fetch_valid_td(start, end):
#     cld = mcal.get_calendar("XSHG")
#     early = cld.schedule(start_date=str(start), end_date=str(end))
#     days = early.index.strftime("%Y%m%d").astype(int)
#     return days[0], days[-1]


class ArgData:
    def __init__(
        self,
        start_time: int = 20190103,
        end_time: int = 20190605,
        max_backtrack_days: int = 100,
        max_future_days: int = 0,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._start_time, self._end_time = start_time, end_time

        self._dates = self._get_data()

    def _get_data(self) -> np.ndarray:
        dates = np.load(DIR_DATES)
        self.start_idx = np.where(dates == self._start_time)[0][0] * MULTI_TI
        self.end_idx = (np.where(dates == self._end_time)[0][0] + 1) * MULTI_TI
        self.total_len = len(dates)
        return dates

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return N_PROD

    @property
    def n_days(self) -> int:
        return (
            self.end_idx
            - self.start_idx
            - self.max_backtrack_days
            - self.max_future_days
        )


def get_feature(
    code,
):
    ret = np.memmap(f"A:/sa/snapshot/{code}.dat", dtype=np.float32, mode="r").reshape(
        -1, len(FEATURES)
    )
    return ret.copy()


def find_date_indices(timestamps, start, end):
    start_idx = np.searchsorted(timestamps, start, side="left")
    end_idx = np.searchsorted(timestamps, end, side="right")
    return start_idx, end_idx


class FakeData:
    def __init__(
        self,
        code="000537",
        start_time=20190103,
        end_time=20190605,
        max_backtrack_days=100,
        max_future_days=0,
        device=torch.device("cpu"),
        **kwargs,
    ):
        self.code = code
        self.device = device

        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.dates = self.load_timestamp_data(code)

        # Find indices
        self.start_idx, self.end_idx = find_date_indices(
            self.dates, start_time, end_time
        )
        # Load feature data within the date range
        self._data = self.load_feature_data(self.code)
        self.label = self.load_label_data(code)

    @property
    def data(self):
        # Transfer the data to the specified device temporarily
        return self._data

    def load_timestamp_data(self, code):
        ret = sa.attach(f"timestamp_{code}", ro=True)
        return ret

    def load_feature_data(self, code):
        ret = np.load(f"/mnt/disk1/snapshot_diff_norm_n20/{code}.npy")
        ret = ret[self.start_idx : self.end_idx, :]
        ret_tensor = torch.from_numpy(ret).float().nan_to_num(0, 0, 0)
        return ret_tensor

    def load_label_data(self, code):
        ret = sa.attach(f"label_{code}", ro=True)
        ret = ret[self.start_idx + self.max_backtrack_days : self.end_idx]
        return torch.from_numpy(ret).float()

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return 1

    @property
    def n_days(self) -> int:
        return (
            self.end_idx
            - self.start_idx
            - self.max_backtrack_days
            - self.max_future_days
        )
