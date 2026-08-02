# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:43:20 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   sklearn.decomposition import PCA

from tqdm import tqdm

class DataPrep:
    
    def __init__(self) -> None: 
        
        self.cur_path    = os.getcwd()
        self.root_path   = os.path.abspath(os.path.join(self.cur_path, ".."))
        self.data_path   = os.path.join(self.root_path, "data")
        self.prep_path   = os.path.join(self.data_path, "PrepData")
        self.ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)
            
        self.vol_target  = 0.1
        self.vol_window  = 100
        self.outlier_rtn = 0.1
        
    def _get_vol_target(
            self, 
            df         : pd.DataFrame, 
            vol_target : float = 0.1, 
            vol_window : int = 100,
            outlier_rtn: float = 0.1) -> pd.DataFrame: 
        
        df_out = (df
                .sort_index()
                .assign(
                    px_rtn   = lambda x: x.PX_LAST.pct_change(),
                    vol      = lambda x: x.px_rtn.ewm(span = vol_window).std() * np.sqrt(252),
                    lag_rtn  = lambda x: x.px_rtn * (vol_target / x.vol.shift()),
                    perf_rtn = lambda x: x.px_rtn * (vol_target / x.vol))
                .assign(
                    lag_rtn  = lambda x: np.where(np.abs(x.lag_rtn) > outlier_rtn, np.nan, x.lag_rtn),
                    perf_rtn = lambda x: np.where(np.abs(x.perf_rtn) > outlier_rtn, np.nan, x.perf_rtn)))
        
        return df_out
            
    def vol_fx_target_returns(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Vol Targeted FX Returns")
        
        out_path = os.path.join(self.prep_path, "FXVolTargetedReturns.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        raw_files = ["FutPX", "FXCarryIndices"]
        df_lists  = []
        
        for file in raw_files: 
            
            in_path  = os.path.join(self.data_path, "RawData", file + ".parquet")
            df_input = (pd
                    .read_parquet(path = in_path, engine = "pyarrow")
                    .set_index("date")
                    .groupby("security")
                    .apply(self._get_vol_target, self.vol_target, self.vol_window, self.outlier_rtn)
                    .reset_index()
                    .melt(id_vars = ["date", "security"]))
            
            df_lists.append(df_input)
            
        if verbose: print("Saving data\n")
        df_out = pd.concat(df_lists)
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def vol_etf_target_returns(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Vol Targeted ETF Returns")
        
        out_path = os.path.join(self.prep_path, "ETFVolTargetedReturns.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path   = os.path.join(self.data_path, "RawData", "EquityPX.parquet")        
        df_out = (pd
                .read_parquet(path = path, engine = "pyarrow")
                .loc[lambda x: x.px_type == "Adj Close"]
                .drop(columns = ["px_type"])
                .set_index("date")
                .rename(columns = {"value": "PX_LAST"})
                .groupby("ticker")
                .apply(self._get_vol_target, self.vol_target, self.vol_window)
                .reset_index())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _calc_rtn_spread(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame: 
        
        df_out = (df.reset_index().melt(
            id_vars    = ["date", ticker],
            var_name   = "for_ticker",
            value_name = "for_rtn").
            dropna().
            rename(columns = {ticker: "dom_rtn"}).
            assign(
                dom_ticker = ticker,
                rtn_spread = lambda x: x.dom_rtn - x.for_rtn))
        
        return df_out
        
    def prepare_eq_rtn(self, verbose: bool = True) -> None: 
        
        etf_path = os.path.join(self.data_path, "ETFPX.parquet")
        out_path = os.path.join(self.data_path, "EquityDifferentials.parquet")
        
        if os.path.exists(out_path) == True: 
            if verbose:
                print("Already have equity returns differential")
                
            return None
        
        if verbose: 
            print("Calculating Returns Differential Data")
        
        df_etf_raw = (pd.read_parquet(
            path = etf_path, engine = "pyarrow").
            assign(date = lambda x: pd.to_datetime(x.Date).dt.date).
            pivot(index = "date", columns = "ticker", values = "Adj Close"))
        
        df_etf_rtn = df_etf_raw.pct_change()
        df_etf_dt  = df_etf_raw.diff().apply(lambda x: x / x.ewm(span = 10, adjust = False).std())
        df_lists   = []
        
        tickers = df_etf_raw.columns.to_list()
        for ticker in tqdm(tickers): 
            
            df_tmp_rtn = (self._calc_rtn_spread(
                df_etf_rtn, ticker).
                assign(rtn_calc = "rtn"))
            
            df_tmp_dt = (self._calc_rtn_spread(
                df_etf_dt, ticker).
                assign(rtn_calc = "dt"))
            
            df_lists.append(df_tmp_rtn)
            df_lists.append(df_tmp_dt)

        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_pcs(self, df: pd.DataFrame, pcs: int = 3) -> pd.DataFrame: 
        
        df_wider = (df
                .pivot(index = "date", columns = "security", values = "PX_LAST")
                .ffill()
                .dropna())
        
        pca_data = PCA(n_components = pcs).fit_transform(df_wider)
        df_pcs   = (pd
                .DataFrame(
                    data    = pca_data,
                    index   = df_wider.index,
                    columns = ["PC{}".format(i + 1) for i in range(pcs)])
                .reset_index()
                .melt(id_vars = "date"))
        
        return df_pcs
        
    def yield_curve_pca(
            self, 
            start_year: int = 2000, 
            pcs       : int = 3,
            verbose   : bool = True) -> None: 
        
        if verbose: print("Getting Yield Curve PCs")
        
        out_path = os.path.join(self.prep_path, "YieldCurvePCs.parquet")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        in_path      = os.path.join(self.data_path, "RawData", "RawYieldCurve.parquet")
        df_curve_raw = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .drop(columns = ["curve"])
                .assign(security = lambda x: x.security.str.split(" ").str[0]))
        
        '''
        df_date_selector = (df_curve_raw
                .drop(columns = ["PX_LAST"])
                .groupby(["security", "country"])
                ["date"]
                .agg("min")
                .to_frame(name = "date")
                .reset_index()
                .groupby(["date", "country"])
                .agg("count")
                .reset_index()
                .sort_values("date")
                .loc[lambda x: x.security >= 5]
                .sort_values("country")
                .drop(columns = ["security"])
                .rename(columns = {"date": "start_date"}))
        
        df_sliced = (df_curve_raw
                .merge(right = df_date_selector, how = "inner", on = ["country"])
                .loc[lambda x: x.date >= x.start_date]
                .drop(columns = ["start_date"]))
        '''
        
        df_date_selector = (df_curve_raw
                .drop(columns = ["PX_LAST"])
                .groupby(["security", "country"])
                ["date"]
                .agg("min")
                .to_frame(name = "date")
                .sort_values("date")
                .assign(year = lambda x: x.date.dt.year)
                .reset_index().
                loc[lambda x: x.year <= start_year]
                .drop(columns = ["date", "year"]))
        
        df_sliced = (df_date_selector
                .merge(right = df_curve_raw, how = "inner", on = ["security", "country"]))

        df_out = (df_sliced
                .groupby("country")
                .apply(self._get_pcs, pcs)
                .reset_index()
                .drop(columns = ["level_1"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    prep_data = DataPrep()
    #prep_data.vol_fx_target_returns()
    #prep_data.vol_etf_target_returns()
    prep_data.yield_curve_pca()
    
if __name__ == "__main__": main()