import SharedArray as sa
import numpy as np
from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

features = ['qopen',
 'qhigh',
 'qlow',
 'qclose',
 'qvolume',
 'qamount',
 'qvwap',
 'qyesterdayclose',
 'qtodayopen',
 'qcumhigh',
 'qcumlow',
 'qcumvolume',
 'qyesterdayvwap',
 'qtwap',
 'qret1',
 'qret5',
 'qret10',
 'qyesterdayskip',
 'q',
 'qintra_midprices_5_mean',
 'qintra_midprices_5_std',
 'qintra_abv_5_mean',
 'qintra_abv_5_std',
 'qintra_deltabuy_5_mean',
 'qintra_deltabuy_5_stdev',
 'qintra_deltasell_5_mean',
 'qintra_deltasell_5_stdev',
 'qintra_deltanetbuy_5_mean',
 'qintra_deltanetbuy_5_stdev',
 'qintra_cumbidval_5_mean',
 'qintra_cumbidval_5_stdev',
 'qintra_cumaskval_5_mean',
 'qintra_cumaskval_5_stdev',
 'qintra_bidheight_5_mean',
 'qintra_askheight_5_mean',
 'qintra_midprice10_ret_mean',
 'qintra_midprice10_ret_stdev',
 'qintra_midp10vol_corr',
 'qintra_price5avgmidp11',
 'qintra_spreadavgmidp11',
 'qintra_midp10riserate_5',
 'qintra_midp10riserate_20',
 'qintra_midp10riserate_100',
 'qintra_tvrriserate_5',
 'qintra_tvrriserate_20',
 'qintra_tvrriserate_100',
 'q',
 'qs_dq_mv',
 'qs_val_mv',
 'qs_val_pe',
 'qs_val_pb_new',
 'qs_val_pe_ttm',
 'qs_val_pcf_ocf',
 'qs_val_pcf_ocfttm',
 'qs_val_pcf_ncf',
 'qs_val_pcf_ncfttm',
 'qs_val_ps',
 'qs_val_ps_ttm',
 'qs_dq_turn',
 'qs_dq_freeturnover',
 'qs_price_div_dps',
 'qnet_profit_parent_comp_ttm',
 'qnet_profit_parent_comp_lyr',
 'qnet_assets_today',
 'qnet_cash_flows_oper_act_ttm',
 'qnet_cash_flows_oper_act_lyr',
 'qoper_rev_ttm',
 'qoper_rev_lyr',
 'qnet_incr_cash_cash_equ_ttm',
 'qnet_incr_cash_cash_equ_lyr',
 'qtot_shr_today',
 'qfloat_a_shr_today',
 'qfree_shares_today',
 'qs_val_pb',
 'qbuy_value_exlarge_order',
 'qsell_value_exlarge_order',
 'qbuy_value_large_order',
 'qsell_value_large_order',
 'qbuy_value_med_order',
 'qsell_value_med_order',
 'qbuy_value_small_order',
 'qsell_value_small_order',
 'qbuy_value_exlarge_order_act',
 'qsell_value_exlarge_order_act',
 'qbuy_value_large_order_act',
 'qsell_value_large_order_act',
 'qbuy_value_med_order_act',
 'qsell_value_med_order_act',
 'qbuy_value_small_order_act',
 'qsell_value_small_order_act',
 'qvalue_diff_small_trader',
 'qvalue_diff_small_trader_act',
 'qvalue_diff_med_trader',
 'qvalue_diff_med_trader_act',
 'qvalue_diff_large_trader',
 'qvalue_diff_large_trader_act',
 'qvalue_diff_institute',
 'qvalue_diff_institute_act',
 'qs_mfd_inflow',
 'qnet_inflow_rate_value',
 'qs_mfd_inflow_open',
 'qopen_net_inflow_rate_value',
 'qs_mfd_inflow_close',
 'qclose_net_inflow_rate_value',
 'qmoneyflow_pct_value',
 'qopen_moneyflow_pct_value',
 'qclose_moneyflow_pct_value',
 'qs_mfd_inflow_large_order',
 'qnet_inflow_rate_value_l',
 'qmoneyflow_pct_value_l',
 'qs_mfd_inflow_open_large_order',
 'qopen_net_inflow_rate_value_l',
 'qopen_moneyflow_pct_value_l',
 'qs_mfd_inflow_close_large_order',
 'qclose_net_inflow_rate_valu_l',
 'qclose_moneyflow_pct_value_l',
 'qind_open',
 'qind_high',
 'qind_low',
 'qind_close',
 'qind_volume',
 'qind_value',
 'qind_ret',
 'qind_vwap',
 'qcne5d_spec_risk',
 'qcne5d_total_risk',
 'qcne5d_hist_beta',
 'qcne5d_pred_beta',
 'qcne5d_spec_ret',
 'qsize_ret_l',
 'qsize_ret_s',
 'qsize_ret_diff',
 'qsize',
 'qcne5d_beta',
 'qcne5d_beta_ret',
 'qcne5d_momentum',
 'qcne5d_momentum_ret',
 'qcne5d_size',
 'qcne5d_size_ret',
 'qcne5d_earnyild',
 'qcne5d_earnyild_ret',
 'qcne5d_resvol',
 'qcne5d_resvol_ret',
 'qcne5d_growth',
 'qcne5d_growth_ret',
 'qcne5d_btop',
 'qcne5d_btop_ret',
 'qcne5d_leverage',
 'qcne5d_leverage_ret',
 'qcne5d_liquidty',
 'qcne5d_liquidty_ret',
 'qcne5d_sizenl',
 'qcne5d_sizenl_ret']
class FeatureType(IntEnum):
    pass

for i, feature in enumerate(features):
    setattr(FeatureType, feature, i)


class StockData:
    def __init__(self,
                 start_time: str,
                 end_time: str,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:1')) -> None:

        self.max_backtrack_days = 100
        self.max_future_days = 30
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._start_time = start_time
        self._end_time = end_time
        self.data, self._dates, self._stock_ids = self._get_data()

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        dates = np.load('basic_info/Dates.npy')
        stock_ids = np.load('basic_info/Univ.npy')
        start_idx = np.where(dates == self._start_time)[0][0]
        end_idx = np.where(dates == self._end_time)[0][0]
        data = []
        for feature in features:
            data.append(sa.attach(feature)[start_idx,end_idx])
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
        columns: Optional[List[str]] = None
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
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)