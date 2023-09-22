from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import SharedArray as sa
import torch
from sklearn.model_selection import train_test_split

from alphagen.dir_config import DIR_DATES

from .feature_list import FEATURES

N_PROD = 6000
MULTI_TI = 16

FeatureType = Enum("FeatureType", {feature: i for i, feature in enumerate(FEATURES)})


def fetch_valid_td(start, end):
    cld = mcal.get_calendar("XSHG")
    early = cld.schedule(start_date=str(start), end_date=str(end))
    days = early.index.strftime("%Y%m%d").astype(int)
    return days[0], days[-1]


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
        self._start_time, self._end_time = fetch_valid_td(start_time, end_time)
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
