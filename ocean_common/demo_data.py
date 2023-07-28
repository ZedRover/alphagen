from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SharedArray as sa
import torch
from feature_list import FEATURES
from enum import Enum

N_PROD = 6000
MULTI_TI = 16

FeatureType = Enum("FeatureType", tuple(FEATURES))


class StockData:
    def __init__(
        self,
        start_time: int = 20190103,
        end_time: int = 20190605,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cuda:1"),
    ) -> None:
        self._instrument = np.load("/home/public2/share_yw/data/basic_info/Univ.npy")
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._start_time = int(start_time)
        self._end_time = int(end_time)
        self.data, self._dates, self._stock_ids = self._get_data()

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ["$" + f.name.lower() for f in self._features]
        dates = np.load("/home/public2/share_yw/data/basic_info/Dates.npy")
        stock_ids = np.load("/home/public2/share_yw/data/basic_info/Univ.npy")
        start_idx = np.where(dates == self._start_time)[0][0]
        end_idx = np.where(dates == self._end_time)[0][0]
        data = []
        for feature in features:
            feature_data = sa.attach(feature[1:])[start_idx:end_idx, ...].reshape(
                -1, N_PROD
            )
            data.append(feature_data)
        data = np.stack(data, axis=1)
        data = torch.from_numpy(data).to(self.device)
        return data, dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `columns`: an optional list of column names
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
