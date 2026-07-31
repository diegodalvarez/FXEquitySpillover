# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:43:20 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   tqdm import tqdm

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

    def prepare_fx_returns(
            self,
            verbose   : bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "FXReturns.parquet")
        
        print(self.data_path)
        return-1
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already Have FX Returns Data")
            return None
                
        if verbose:
            print("Collecting FX Returns Data")
        
        df_ticker = self.df_ticker[["ticker", "group"]]
        
        df_fut_ticker = (df_ticker.query(
            "group == ['illiquid_future', 'liquid_future']").
            rename(columns = {"ticker": "security"}))
        
        df_carry_ticker = (df_ticker.query(
            "group == ['em_carry', 'g10_carry']").
            rename(columns = {"ticker": "security"}))
        
        df_fut = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            drop_duplicates().
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = "date", value_name = "rtn").
            dropna().
            merge(right = df_fut_ticker, how = "inner", on = ["security"]))
        
        df_carry = (pd.read_parquet(
            path = carry_path, engine = "pyarrow").
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = "date", value_name = "rtn").
            dropna().
            merge(right = df_carry_ticker, how = "inner", on = ["security"]))
        
        df_out = pd.concat([df_fut, df_carry])
        if verbose: 
            print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def prepare_raw_factor(self, path: str, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "EquityFactors.parquet")
        print(out_path)
        
        if os.path.exists(out_path) == True: 
            if verbose:
                print("Already have Equity Factor Data")
            return None
        
        if verbose:
            print("Getting Equity Factor data")
        
        best_paths = [
            os.path.join(path, group + "Best.parquet")
            for group in self.etf_groups]
        
        real_paths = [
            os.path.join(path, group + "Real.parquet")
            for group in self.etf_groups]
        
        df_tmp_ticker = (self.df_ticker[
            ["country", "etf_benchmark", "etf_ticker"]].
            rename(columns = {"etf_benchmark": "security"}).
            dropna().
            drop_duplicates())
        
        est_namer = (self.df_variable.set_index(
            "estimated").
            tmp_name.
            to_dict())
        
        real_namer = (self.df_variable.set_index(
            "real").
            tmp_name.
            to_dict())
        
        df_best = (pd.read_parquet(
            path = best_paths, engine = "pyarrow").
            drop(columns = ["BEST_TARGET_PRICE"]).
            melt(
                id_vars    = ["date", "security"],
                var_name   = "best_ticker",
                value_name = "best_val").
            drop_duplicates().
            groupby(["date", "security", "best_ticker"]).
            agg("mean").
            reset_index() .
             merge(right = df_tmp_ticker, how = "inner", on = ["security"]).
             assign(factor = lambda x: x.best_ticker.map(est_namer)).
             drop(columns = ["best_ticker"]))

        df_real = (pd.read_parquet(
            path = real_paths, engine = "pyarrow").
            assign(EV_EBITDA = lambda x: x.CURR_ENTP_VAL / x.EBITDA). 
            # ^ Since EV/EBITDA isn't readily available for real calculation
            melt(
                id_vars    = ["date", "security"],
                var_name   = "real_ticker",
                value_name = "real_val").
            drop_duplicates().
            groupby(["date", "security", "real_ticker"]).
            agg("mean").
            reset_index().
            dropna().
            merge(right = df_tmp_ticker, how = "inner", on = ["security"]).
            assign(factor = lambda x: x.real_ticker.map(real_namer)).
            drop(columns = ["real_ticker"]).
            dropna())
        
        df_out = (df_best.merge(
            right = df_real, 
            how   = "inner", 
            on    = ["date", "security", "etf_ticker", "factor", "country"]))
        
        if verbose:
            print("Saving Equity Factor Data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    prep_data = DataPrep()
    prep_data.vol_fx_target_returns()
    prep_data.vol_etf_target_returns()
    
    #setup.prepare_eq_rtn()
    
    #path = r"A:\2026BlpAdHocData\April2026"
    #setup.prepare_raw_factor(path)
    
if __name__ == "__main__": main()