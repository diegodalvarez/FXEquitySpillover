# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:55:34 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

class SignalOptimizer:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.opt_path  = os.path.join(self.data_path, "OptimizedSignal")
        
        if not os.path.exists(self.opt_path):
            os.makedirs(self.opt_path)
            
        self.signals = ["ReturnSpreadSignal", "ReturnsSignal"]
        self.q       = 10
        
    def _insample_optimize(self, df: pd.DataFrame, q: int = 10) -> pd.DataFrame: 
        
        df_decile = (df
                .sort_index()
                .assign(
                    decile     = lambda x: pd.qcut(x = x.signal, q = q, labels = [i + 1 for i in range(q)]),
                    lag_decile = lambda x: x.decile.shift())
                .reset_index())
        
        df_sharpe = (df_decile
                [["lag_decile", "fx_rtn"]]
                .reset_index(drop = True)
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252)))
        
        df_tmp_decile = (df_sharpe
                .reset_index()
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(group = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_decile
                .drop(columns = ["lag_decile"])
                .groupby("group")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x.fx_rtn > 0, 1, 0))
                .drop(columns = ["fx_rtn"])
                .merge(right = df_tmp_decile, how = "outer", on = ["group"])
                .rename(columns = {"fx_rtn": "sharpe"})
                .merge(right = df_decile, how = "outer", on = ["lag_decile"]))
        
        return df_out
    
    '''
    def _outsample_optimize(self, df: pd.DataFrame, q: int, min_obs: int = 5) -> pd.DataFrame: 
        
        df = df.sort_index()
    
        opt_dates = (df
            .index
            .to_series()
            .groupby(pd.Grouper(freq="W-FRI"))
            .max()
            .dropna()
            .to_list()
            [min_obs:])
        
        df_out = []
    
        for opt_date in tqdm(opt_dates):
            
            df_insample = df.loc[:opt_date].copy()
            
            df_insample["decile"], bins = pd.qcut(
                x       = df_insample["signal"],
                q       = q,
                labels  = False,
                retbins = True)
            
            df_insample["decile"]    += 1
            df_insample               = df_insample.sort_values("date")
            df_insample["lag_decile"] = df_insample["decile"].shift()
            
            df_sharpe = (df_insample
                    [["lag_decile", "fx_rtn"]]
                    .reset_index(drop = True)
                    .groupby("lag_decile")
                    .agg(lambda x: x.mean() / x.std() * np.sqrt(252))
                    .reset_index()
                    .rename(columns = {"fx_rtn": "sharpe"}))
            
            df_tmp_decile = (df_sharpe
                    .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                    .assign(group = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
            
            df_signal_scaler = (df_tmp_decile
                    .drop(columns = ["lag_decile"])
                    .groupby("group")
                    .agg("prod")
                    .assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, 0)))
            
            df_decile_add = (df_signal_scaler
                    .reset_index()
                    .drop(columns = ["sharpe"])
                    .merge(right = df_tmp_decile, how = "inner", on = ["group"]))
            
            df_oos = (df
                    .reset_index()
                    .loc[lambda x: (opt_date < x.date) & (x.date <= opt_date + pd.Timedelta(days = 7))]
                    .copy())
            
            df_tmp_opt = (df_oos
                    .assign(
                        decile     = lambda x: pd.cut(x.signal, bins = bins, labels = False) + 1,
                        lag_decile = lambda x: x.decile.shift())
                    .merge(right = df_decile_add, how = "left", on = ["lag_decile"]))
            
            df_out.append(df_tmp_opt)
    
        df_out = pd.concat(df_out)
        return df_out
    '''
    def _outsample_optimize(
        self,
        df     : pd.DataFrame,
        q      : int,
        min_obs: int = 5) -> pd.DataFrame:
    
        df = df.sort_index()
    
        opt_dates = (
            df.index
            .to_series()
            .groupby(pd.Grouper(freq="W-FRI"))
            .max()
            .dropna()
            .to_list())[min_obs:]
    
        df_out = []
    
        for opt_date in tqdm(opt_dates):
    
            # -------------------------
            # In-sample
            # -------------------------
    
            df_is = df.loc[:opt_date, ["signal", "fx_rtn"]].copy()
    
            # qcut once
            df_is["decile"], bins = pd.qcut(
                df_is["signal"],
                q=q,
                labels=False,
                retbins=True
            )
    
            df_is["decile"] += 1
    
            # Lag the decile
            df_is["lag_decile"] = df_is["decile"].shift()
    
            # -------------------------
            # Sharpe by decile
            # -------------------------
    
            df_sharpe = (
                df_is
                .dropna(subset=["lag_decile"])
                .groupby("lag_decile")["fx_rtn"]
                .agg(["mean", "std"])
                .assign(
                    sharpe=lambda x:
                        x["mean"] / x["std"] * np.sqrt(252)
                )
                .reset_index()
                [["lag_decile", "sharpe"]]
            )
    
            # Only extreme deciles
            df_tmp_decile = (
                df_sharpe
                .loc[lambda x: x["lag_decile"].isin([1, 2, 9, 10])]
                .assign(
                    group=lambda x: np.where(
                        x["lag_decile"] <= 2,
                        "lgroup",
                        "ugroup"
                    )
                )
            )
    
            # -------------------------
            # Determine scaler
            # -------------------------
    
            df_signal_scaler = (
                df_tmp_decile
                .groupby("group")["sharpe"]
                .prod()
                .gt(0)
                .astype(int)
                .rename("signal_scaler")
            )
    
            # -------------------------
            # Out-of-sample
            # -------------------------
    
            df_oos = df.loc[
                (df.index > opt_date) &
                (df.index <= opt_date + pd.Timedelta(days=7))
            ].copy()
    
            # Apply IS bins to OOS observations
            df_oos["decile"] = (
                pd.cut(
                    df_oos["signal"],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                ) + 1
            )
    
            # Lag decile exactly as before
            df_oos["lag_decile"] = df_oos["decile"].shift()
            
            df_oos = df_oos.reset_index().merge(
                df_tmp_decile[
                    ["lag_decile", "sharpe", "group"]
                ],
                how="left",
                on="lag_decile"
            )
    
            # Map lgroup / ugroup to OOS deciles
            df_oos["group"] = np.select(
                [
                    df_oos["lag_decile"].isin([1, 2]),
                    df_oos["lag_decile"].isin([9, 10])
                ],
                [
                    "lgroup",
                    "ugroup"
                ],
                default=None
            )
    
            # Map scaler
            df_oos["signal_scaler"] = (
                df_oos["group"]
                .map(df_signal_scaler)
            )
    
            df_oos["opt_date"] = opt_date
    
            df_out.append(df_oos)
    
        return pd.concat(df_out)
        
    def _optimize_signal(self, df: pd.DataFrame, q: int, verbose: bool = True) -> pd.DataFrame: 
        
        if verbose: print("Working on {}".format(df.name))
        
        df_is = (self
                 ._insample_optimize(df, q)
                 .assign(
                     opt_date     = np.nan,
                     sample_group = "in_sample"))
        
        df_os = (self
                 ._outsample_optimize(df, q)
                 .assign(sample_group = "out_sample"))
        
        df_out = pd.concat([df_is, df_os])
        if verbose: print("\n")
        return df_out
        
    def optimize_signal(self, verbose: bool = True) -> None: 
        
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        df_rtn   = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.variable == "perf_rtn"]
                .drop(columns = ["variable"])
                .dropna()
                .rename(columns = {
                    "security": "fx_ticker",
                    "value"   : "fx_rtn"}))
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                [["ticker", "etf_ticker"]]
                .dropna()
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        for signal in self.signals:
            
            in_path  = os.path.join(self.data_path, "Signals", signal + ".parquet")
            out_path = os.path.join(self.opt_path, signal + ".parquet")
            
            if os.path.exists(out_path):
                if verbose: print("Already have {}\n".format(signal))
                continue
            
            df_signal = (pd
                    .read_parquet(path = in_path, engine = "pyarrow")
                    [["date", "variable", "ticker", "signal", "window"]]
                    .loc[lambda x: x.variable.isin(["perf_rtn", "px_rtn"])]
                    .rename(columns = {"ticker": "etf_ticker"})
                    .merge(right = df_ticker, how = "inner", on = ["etf_ticker"])
                    .merge(right = df_rtn, how = "inner", on = ["date", "fx_ticker"])
                    .rename(columns = {"variable": "signal_type"})
                    .assign(group_var = lambda x: x.signal_type + "-" + x.etf_ticker + "-" + x.fx_ticker)
                    .set_index("date")
                    .groupby("group_var")
                    .apply(self._optimize_signal, self.q)
                    .reset_index()
                    .drop(columns = ["level_1"])
                    .assign(signal = signal))
            
            if verbose: print("Saving data\n")
            df_signal.to_parquet(path = out_path, engine = "pyarrow")

signal_optimizer = SignalOptimizer()
signal_optimizer.optimize_signal()