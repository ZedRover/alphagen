from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SharedArray as sa
import torch
from .feature_list import FEATURES
from enum import Enum
import pandas_market_calendars as mcal

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
        self._instrument = np.load("/home/public2/share_yw/data/basic_info/Univ.npy")
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._start_time, self._end_time = fetch_valid_td(start_time, end_time)
        self.data, self._dates, self._stock_ids = self._get_data()

    def _get_data(self) -> Tuple[None, np.ndarray, np.ndarray]:
        dates = np.load("/home/public2/share_yw/data/basic_info/Dates.npy")
        stock_ids = np.load("/home/public2/share_yw/data/basic_info/Univ.npy")
        self.start_idx = np.where(dates == self._start_time)[0][0] * MULTI_TI
        self.end_idx = np.where(dates == self._end_time)[0][0] * MULTI_TI

        return None, dates, stock_ids

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

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `columns`: an optional list of column names.
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"number of days in the provided tensor ({n_days}) doesn't "
                f"match that of the current StockData ({self.n_days})"
            )
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                f"match that of the current StockData ({self.n_stocks})"
            )
        if len(columns) != n_columns:
            raise ValueError(
                f"size of columns ({len(columns)}) doesn't match with "
                f"tensor feature count ({data.shape[2]})"
            )
        date_index = self._dates[self.max_backtrack_days : -self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
