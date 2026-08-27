# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:55:34 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        
        self.q          = 10
        self.slice_year = 2018
        
    def _insample_optimize(self, df: pd.DataFrame, rtn_name: str, q: int = 10) -> pd.DataFrame: 
        
        df_decile = (df
                .sort_index()
                .assign(
                    decile     = lambda x: pd.qcut(x = x.signal, q = q, labels = [i + 1 for i in range(q)]),
                    lag_decile = lambda x: x.decile.shift())
                .reset_index())
        
        df_sharpe = (df_decile
                [["lag_decile", rtn_name]]
                .reset_index(drop = True)
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252)))
        
        df_tmp_decile = (df_sharpe
                .reset_index()
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(dgroup = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_decile
                .drop(columns = ["lag_decile"])
                .groupby("dgroup")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x[rtn_name] > 0, 1, 0))
                .drop(columns = [rtn_name])
                .merge(right = df_tmp_decile, how = "outer", on = ["dgroup"])
                .rename(columns = {rtn_name: "sharpe"})
                .merge(right = df_decile, how = "outer", on = ["lag_decile"]))
        
        return df_out
    
    def _train_test_optimize(
            self, 
            df        : pd.DataFrame, 
            rtn_name  : str, 
            slice_year: int = 2018, 
            q         : int = 10) -> pd.DataFrame: 
        
        df_train_test = (df
                .assign(
                    year         = lambda x: x.date.dt.year,
                    sample_group = lambda x: np.where(x.year <= slice_year, "in_sample", "out_sample"))
                .drop(columns = ["year"]))
        
        opt_date = (df_train_test
                .loc[lambda x: x.sample_group == "in_sample"]
                .date
                .max())
        
        df_insample = (df_train_test
                .loc[lambda x: x.sample_group == "in_sample"])
        
        _, bins = (pd
                   .qcut(
                       x       = df_insample.signal,
                       q       = q,
                       labels  = [i + 1 for i in range(q)],
                       retbins = True))
        
        bins[0], bins[-1] = -np.inf, np.inf
        
        df_decile = (df_train_test
                .assign(decile = lambda x: pd
                        .cut(
                            x      = x.signal, 
                            bins   = bins,
                            labels = range(1,q+1)),
                        lag_decile = lambda x: x.decile.shift()))
        
        df_sharpe = (df_decile
                .loc[lambda x: x.sample_group == "in_sample"]
                [["lag_decile", "vol_fx_rtn"]]
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252))
                .rename(columns = {"vol_fx_rtn": "sharpe"})
                .reset_index())
        
        df_tmp_sharpe = (df_sharpe
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(dgroup = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_sharpe
                .drop(columns = ["lag_decile"])
                .groupby("dgroup")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, np.nan))
                .drop(columns = ["sharpe"])
                .reset_index()
                .merge(right = df_tmp_sharpe, how = "inner", on = ["dgroup"])
                .merge(right = df_decile, how = "outer", on = ["lag_decile"])
                .assign(opt_date = opt_date))
        
        return df_out
    
    def _outsample_optimize(
        self,
        df      : pd.DataFrame,
        rtn_name: str,
        q       : int = 10,
        min_obs : int = 5,
        verbose : bool = True,
        debug   : bool = False) -> pd.DataFrame:
    
        df = df.sort_index()
        
        opt_dates = (
            df.index
            .to_series()
            .groupby(pd.Grouper(freq="W-FRI"))
            .max()
            .dropna()
            .to_list())[min_obs:]
    
        if debug: 
            print("[WARNING] Running Debugging")
            opt_dates = opt_dates[0:5]
    
        df_out = []
        
        if verbose: iterable = tqdm(opt_dates)
        else      : iterable = opt_dates
    
        for opt_date in iterable:
    
            # -------------------------
            # In-sample
            # -------------------------
    
            df_is = df.loc[:opt_date, ["signal", rtn_name]].copy()
    
            # qcut once
            df_is["decile"], bins = pd.qcut(
                x       = df_is["signal"],
                q       = q,
                labels  = False,
                retbins = True
            )
    
            df_is["decile"]     += 1
            df_is["lag_decile"]  = df_is["decile"].shift()
    
            # -------------------------
            # Sharpe by decile
            # -------------------------
    
            df_sharpe = (
                df_is
                .dropna(subset=["lag_decile"])
                .groupby("lag_decile")[rtn_name]
                .agg(["mean", "std"])
                .assign(sharpe=lambda x: x["mean"] / x["std"] * np.sqrt(252))
                .reset_index()
                [["lag_decile", "sharpe"]])
    
            # Only extreme deciles
            df_tmp_decile = (
                df_sharpe
                .loc[lambda x: x["lag_decile"].isin([1, 2, 9, 10])]
                .assign(dgroup=lambda x: np.where(x["lag_decile"] <= 2, "lgroup", "ugroup")))
    
            # -------------------------
            # Determine scaler
            # -------------------------
    
            df_signal_scaler = (
                df_tmp_decile
                .groupby("dgroup")["sharpe"]
                .prod()
                .gt(0)
                .astype(int)
                .rename("signal_scaler"))
    
            # -------------------------
            # Out-of-sample
            # -------------------------
    
            df_oos = (df
                      .loc[(df.index > opt_date) & (df.index <= opt_date + pd.Timedelta(days=7))]
                      .copy())
    
            # Apply IS bins to OOS observations
            df_oos["decile"] = (
                pd.cut(
                    x              = df_oos["signal"],
                    bins           = bins,
                    labels         = False,
                    include_lowest = True) + 1 )
    
            # Lag decile exactly as before
            df_oos["lag_decile"] = df_oos["decile"].shift()
            
            df_oos = df_oos.reset_index().merge(
                df_tmp_decile[
                    ["lag_decile", "sharpe", "dgroup"]
                ],
                how="left",
                on="lag_decile"
            )
    
            # Map lgroup / ugroup to OOS deciles
            df_oos["dgroup"] = np.select(
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
                df_oos["dgroup"]
                .map(df_signal_scaler)
            )
    
            df_oos["opt_date"] = opt_date
    
            df_out.append(df_oos)
    
        return pd.concat(df_out)
        
    def _optimize_signal(
            self, 
            df        : pd.DataFrame, 
            slice_year: int,
            q         : int, 
            verbose   : bool = True) -> pd.DataFrame: 
        
        if verbose: print("Working on {}".format(df.name))
        
        df_is = (self
                 ._insample_optimize(df, "vol_fx_rtn", q)        
                 .assign(
                    opt_date     = np.nan,
                    optimization = "in_sample",
                    sample_group = "in_sample")
                 .drop(columns = ["index"]))
        
        df_train_test = (self
                ._train_test_optimize(df, "vol_fx_rtn", slice_year, q)
                .assign(optimization = "train_test"))
        
        df_expanding = (self
                 ._outsample_optimize(df.set_index("date"), "vol_fx_rtn", q, 30, verbose)
                 .assign(
                     optimization = "expanding",
                     sample_group = "out_sample"))
    
        df_out = pd.concat([df_is, df_train_test, df_expanding])
        if verbose: print("\n")
        return df_out
        
    def optimize_spread_trend(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Optimized Trend")
        
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        sig_path = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        out_path = os.path.join(self.data_path, "OptimizedSignal", "OptimizedTrendSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {
                    "security": "fx_ticker",
                    "fx_rtn"  : "vol_fx_rtn"}))
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .loc[lambda x: x.Active == True]
                .drop(columns = [
                    "name", "country", "Active", "etf_benchmark", "FACTOR_ACTIVE",
                    "CurveFile"])
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        df_out = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["lag_signal", "window", "signal_name"])
                .rename(columns = {"ticker": "etf_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["etf_ticker"])
                .merge(right = df_fx_rtn, how = "inner", on = ["date", "fx_ticker"])
                .assign(
                    date      = lambda x: pd.to_datetime(x.date),
                    group_var = lambda x: x.fx_ticker + "-" + x.etf_ticker + "-" + x.group)
                #.loc[lambda x: x.group_var == x.group_var.min()]
                .groupby("group_var")
                .apply(self._optimize_signal, self.slice_year, self.q, verbose)
                .reset_index(drop = True))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def optimize_ols_resid(self, year: int = 2018, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Optimized Trend")
        
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        model_path  = os.path.join(self.data_path, "Signals", "ResidOLS.pkl")
        signal_path = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        out_path = os.path.join(self.data_path, "OptimizedSignal", "OptimizedOLSResidual.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_signal = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["lag_signal", "window", "signal_name"])
                .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {
                    "security": "fx_ticker",
                    "fx_rtn"  : "vol_fx_rtn"}))
        
        with open(model_path, "rb") as f:
            trend_models = pickle.load(f)
        
        if verbose: iterable = tqdm(trend_models.keys())
        else      : iterable = trend_models.keys()
        
        df_lists      = []
        df_raw_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        for key in iterable:
            
            tmp_model             = trend_models[key]
            fx_ticker, etf_ticker = key.split("-")
            
            df_tmp_fx = (df_fx_rtn
                    .loc[lambda x: x.fx_ticker == fx_ticker]
                    .set_index("date"))
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.etf_ticker == etf_ticker]
                    .set_index("date")
                    .dropna())
            
            # full-sample
            df_full = (tmp_model
                    ["full"]
                    .resid
                    .to_frame(name = "signal")
                    .merge(right = df_tmp_fx, how = "inner", on = ["date"])
                    .pipe(lambda x: self._insample_optimize(x, "vol_fx_rtn"))
                    .assign(
                        opt_date     = lambda x: x.date.max(),
                        regression   = "full_sample",
                        sample_group = "in_sample",
                        optimization = "full_sample"))
            
            # train/test
            df_train = (tmp_model
                    ["train"]
                    .predict(sm.add_constant(df_tmp_signal.signal))
                    .to_frame(name = "pred")
                    .merge(right = df_tmp_fx, how = "inner", on = ["date"])
                    .reset_index()
                    .assign(
                        resid        = lambda x: x.vol_fx_rtn - x.pred,
                        year         = lambda x: x.date.dt.year,
                        regression   = "train_test",
                        sample_group = lambda x: np.where(x.year <= year, "in_sample", "out_sample"))
                    .drop(columns = ["year", "pred"])
                    .rename(columns = {"resid": "signal"})
                    .pipe(lambda x: self._train_test_optimize(x, year))
                    .assign(
                        regression   = "train_test",
                        optimization = "train_test"))
        
            # for expanding
            df_expanding = (tmp_model
                    ["expanding"]
                    .params
                    .rename(columns = {"signal": "beta"})
                    .shift()
                    .merge(right = df_tmp_signal, how = "inner", on = ["date"])
                    .merge(right = df_tmp_fx,     how = "inner", on = ["date"])
                    .assign(
                        pred         = lambda x: (x.beta * x.signal) + x.const,
                        resid        = lambda x: x.vol_fx_rtn - x.pred,
                        regression   = "expanding",
                        sample_group = "out_sample")
                    .drop(columns = ["const", "beta", "etf_ticker", "pred", "signal"])
                    .rename(columns = {"resid": "signal"})
                    .dropna()
                    .pipe(lambda x: self._outsample_optimize(x, "vol_fx_rtn"))
                    .assign(
                        optimization = "expanding",
                        sample_group = "out_sample"))

            df_add = (pd
                    .concat([df_full, df_train, df_expanding])
                    .assign(
                        signal_name = key,
                        fx_ticker   = fx_ticker,
                        etf_ticker  = etf_ticker)
                    .merge(right = df_raw_fx_rtn, how = "inner", on = ["date", "fx_ticker"]))

            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
        return-1
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .loc[lambda x: x.Active == True]
                .drop(columns = [
                    "name", "country", "Active", "etf_benchmark", "FACTOR_ACTIVE",
                    "CurveFile"])
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        df_out = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["lag_signal", "window", "signal_name"])
                .rename(columns = {"ticker": "etf_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["etf_ticker"])
                .merge(right = df_fx_rtn, how = "inner", on = ["date", "fx_ticker"])
                .assign(
                    date      = lambda x: pd.to_datetime(x.date),
                    group_var = lambda x: x.fx_ticker + "-" + x.etf_ticker + "-" + x.group)
                #.loc[lambda x: x.group_var == x.group_var.min()]
                .groupby("group_var")
                .apply(self._optimize_signal, self.slice_year, self.q, verbose)
                .reset_index(drop = True))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None:

    signal_optimizer = SignalOptimizer()
    #signal_optimizer.optimize_spread_trend(verbose = True)
    signal_optimizer.optimize_ols_resid(verbose = True)

if __name__ == "__main__": main()