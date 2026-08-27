# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:58:37 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
tqdm.pandas()

class Backtesting:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.back_path = os.path.join(self.data_path, "Backtests")
        
        if not os.path.exists(self.back_path):
            os.makedirs(self.back_path)
            
        self.vol_target = 0.1
        self.vol_window = 100
        self.threshold  = 0.1
        
    def _vol_hedge(
            self, 
            df        : pd.DataFrame, 
            vol_target: float = 0.1,
            vol_window: int   = 100,
            threshold : float = 0.1) -> pd.DataFrame: 
        
        df_perf = (df
                .reset_index()
                .pivot(index = "date", columns = "name", values = "signal_rtn")
                .apply(lambda x: x * (vol_target / (x.ewm(span = vol_window, adjust = False).std() * np.sqrt(252))))
                .apply(lambda x: np.where(np.abs(x) > threshold, np.nan, x))
                .reset_index()
                .melt(id_vars = "date", value_name = "vol_rtn"))
        
        df_lagged = (df
                .reset_index()
                .pivot(index = "date", columns = "name", values = "signal_rtn")
                .apply(lambda x: x * (vol_target / (x.ewm(span = vol_window, adjust = False).std().shift() * np.sqrt(252))))
                .apply(lambda x: np.where(np.abs(x) > threshold, np.nan, x))
                .reset_index()
                .melt(id_vars = "date", value_name = "lag_vol_rtn"))
        
        df_out = (df
                .merge(right = df_lagged, how = "inner", on = ["date", "name"])
                .merge(right = df_perf  , how = "inner", on = ["date", "name"]))
        
        return df_out
        
    def _get_port_rtn(self, df: pd.DataFrame, vol_target: float, vol_window: int, threshold: float) -> pd.DataFrame: 
        
        df_signal_rtn = (df
                .assign(signal_rtn = lambda x: np.sign(x.sharpe) * x.signal_scaler * x.fx_rtn))
        
        df_port_wider = (df_signal_rtn
                         .pivot(
                             index   = "date", 
                             columns = "fx_ticker", 
                             values  = "signal_rtn"))
        
        df_port_out = (self
                       ._vol_target_rtn(df_port_wider, vol_target, vol_window, threshold)
                       .assign(portfolio = "full"))
        
        groups   = df.group.drop_duplicates().sort_values().to_list()
        df_lists = []
        
        for group in groups: 
            
            df_signal_wider = (df_signal_rtn
                    .loc[lambda x: x.group == group]
                    .pivot(index = "date", columns = "fx_ticker", values = "signal_rtn"))
            
            df_add = (self
                    ._vol_target_rtn(df_signal_wider, vol_target, vol_window, threshold)
                    .assign(portfolio = group))
            
            df_lists.append(df_add)
            
        df_lists.append(df_port_out)
        df_out = pd.concat(df_lists)
        
        return df_out
    
    def get_spread_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting signal backtests")
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        sig_path    = os.path.join(self.data_path, "Signals", "ReturnSpreadSignal.parquet")
        out_path    = os.path.join(self.back_path, "SignalBacktests.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have backtested signal backtests\n")
            return None
        
        df_ticker = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .drop(columns = ["CurveFile", "name", "etf_benchmark"])
                .loc[lambda x: x.Active == True]
                .dropna()
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
                .rename(columns = {"ticker": "fx_ticker"}))
    
        df_signal = (pd
              .read_parquet(path = sig_path)
              .set_index("date")
              [["ticker", "spread"]]
              .groupby("ticker")
              .apply(lambda x: x.sort_index().spread.shift())
              .reset_index()
              .rename(columns = {
                  "spread": "lag_spread",
                  "ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_prep = (df_signal
                .merge(right = df_ticker, how = "inner", on = ["etf_ticker"])
                .merge(right = df_fx_rtn, how = "inner", on = ["fx_ticker", "date"])
                .assign(
                    fx_rtn     = lambda x: np.where(x.return_type == "carry", -x.fx_rtn, x.fx_rtn),
                    signal_rtn = lambda x: np.sign(x.lag_spread) * x.fx_rtn,
                    name       = lambda x: x.fx_ticker + " " + x.etf_ticker)
                .set_index("date"))
        
        df_out = self._vol_hedge(df_prep, self.vol_target, self.vol_window, self.threshold)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_spread_trend_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting trend signal backtests")
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        sig_path    = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        out_path    = os.path.join(self.back_path, "TrendSignalBacktests.parquet")
    
        if os.path.exists(out_path):
            if verbose: print("Already have backtested signal backtests\n")
            return None
        
        df_ticker = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .drop(columns = ["CurveFile", "name", "etf_benchmark"])
                .loc[lambda x: x.Active == True]
                .dropna()
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
                .rename(columns = {"ticker": "fx_ticker"}))
    
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["signal", "window", "signal_name"])
                .dropna()
                .rename(columns = {
                    "spread": "lag_trend",
                    "ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_prep = (df_signal
                .merge(right = df_ticker, how = "inner", on = ["etf_ticker"])
                .merge(right = df_fx_rtn, how = "inner", on = ["fx_ticker", "date"])
                .assign(
                    fx_rtn     = lambda x: np.where(x.return_type == "carry", -x.fx_rtn, x.fx_rtn),
                    signal_rtn = lambda x: np.sign(x.lag_signal) * x.fx_rtn,
                    name       = lambda x: x.fx_ticker + " " + x.etf_ticker)
                .set_index("date"))
        
        df_out = self._vol_hedge(df_prep, self.vol_target, self.vol_window, self.threshold)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_forecasted_ols_backtest(self, verbose: bool = True, year: int = 2018) -> None: 
        
        if verbose: print("Getting Forecasted Backtest")
        
        out_path = os.path.join(self.back_path, "ForecastedBacktest.parquet")        
        
        if os.path.exists(out_path):
            if verbose: print("Already have forecasted Backtests\n")
            return None
        
        model_path = os.path.join(self.data_path, "Signals", "ForecastedOLS.pkl")
        with open(model_path, "rb") as f: trend_models = pickle.load(f)
        
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .rename(columns = {"security": "fx_ticker"})
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"]))
        
        signal_path = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        df_signal   = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["signal", "window", "signal_name"]))
        
        df_lists = []
        
        if verbose: iterable = tqdm(trend_models.keys())
        else      : iterable = trend_models.keys()
        
        for key in iterable:
            
            tmp_model             = trend_models[key]
            fx_ticker, etf_ticker = key.split("-")
            
            df_tmp_fx = (df_fx_rtn
                    .loc[lambda x: x.fx_ticker == fx_ticker]
                    .set_index("date"))
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.ticker == etf_ticker]
                    .set_index("date")
                    .dropna())

            # full-sample
            df_full = (tmp_model
                    ["full"]
                    .fittedvalues
                    .to_frame(name = "forecasted")
                    .assign(
                        regression   = "full_sample",
                        sample_group = "in_sample"))
            
            # train/test
            df_train = (tmp_model
                    ["train"]
                    .predict(sm.add_constant(df_tmp_signal.lag_signal))
                    .to_frame(name = "forecasted")
                    .assign(regression = "in_sample")
                    .reset_index()
                    .assign(
                        year = lambda x: x.date.dt.year,
                        regression   = "train_test",
                        sample_group = lambda x: np.where(x.year <= year, "in_sample", "out_sample"))
                    .drop(columns = ["year"])
                    .set_index("date"))
            
            # for expanding
            df_expanding = (tmp_model
                    ["expanding"]
                    .params
                    .rename(columns = {"lag_signal": "beta"})
                    .shift()
                    .merge(right = df_tmp_signal, how = "inner", on = ["date"])
                    .assign(
                        regression   = "expanding",
                        sample_group = "out_sample",
                        forecasted   = lambda x: np.sign(x.beta * x.lag_signal) + x.const)
                    [["forecasted", "sample_group", "regression"]])

            df_add = (pd
                    .concat([df_full, df_train, df_expanding])
                    .assign(
                        signal_name = key,
                        fx_ticker   = fx_ticker,
                        etf_ticker  = etf_ticker)
                    .merge(right = df_tmp_fx, how = "inner", on = ["date", "fx_ticker"]))

            df_lists.append(df_add)
            
        df_prep = (pd
                .concat(df_lists)
                .assign(
                    signal_rtn = lambda x: np.sign(x.forecasted) * x.fx_rtn,
                    name       = lambda x: x.regression + " " + x.fx_ticker + " " + etf_ticker))
        
        df_out = self._vol_hedge(df_prep, self.vol_target, self.vol_window, self.threshold)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
            
    def get_residual_ols_backtest(self, verbose: bool = True, year: int = 2018) -> None: 
        
        if verbose: print("Getting Residual Backtest")
        
        out_path = os.path.join(self.back_path, "ResidualBacktest.parquet")
        if os.path.exists(out_path):
            if verbose: print("Already have forecasted Backtests\n")
            return None
        
        model_path = os.path.join(self.data_path, "Signals", "ResidOLS.pkl")
        with open(model_path, "rb") as f: trend_models = pickle.load(f)
        
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .rename(columns = {"security": "fx_ticker"})
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"]))
        
        signal_path = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        df_signal   = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["lag_signal", "window", "signal_name"]))
        
        df_lists = []
        
        if verbose: iterable = tqdm(trend_models.keys())
        else      : iterable = trend_models.keys()
        
        for key in iterable:
            
            tmp_model             = trend_models[key]
            fx_ticker, etf_ticker = key.split("-")
            
            df_tmp_fx = (df_fx_rtn
                    .loc[lambda x: x.fx_ticker == fx_ticker]
                    .set_index("date"))
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.ticker == etf_ticker]
                    .set_index("date")
                    .dropna())
            
            # full-sample
            df_full = (tmp_model
                    ["full"]
                    .resid
                    .to_frame(name = "resid")
                    .assign(
                        regression   = "full_sample",
                        sample_group = "in_sample",
                        lag_resid    = lambda x: x.resid.shift()))
            
            # train/test
            df_train = (tmp_model
                    ["train"]
                    .predict(sm.add_constant(df_tmp_signal.signal))
                    .to_frame(name = "pred")
                    .merge(right = df_tmp_fx, how = "inner", on = ["date"])
                    .reset_index()
                    .assign(
                        resid        = lambda x: x.fx_rtn - x.pred,
                        lag_resid    = lambda x: x.resid.shift(),
                        year         = lambda x: x.date.dt.year,
                        regression   = "train_test",
                        sample_group = lambda x: np.where(x.year <= year, "in_sample", "out_sample"))
                    .drop(columns = ["year", "fx_rtn", "fx_ticker", "pred"])
                    .set_index("date"))
            
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
                        resid        = lambda x: x.fx_rtn - x.pred,
                        lag_resid    = lambda x: x.resid.shift(),
                        regression   = "expanding",
                        sample_group = "out_sample")
                    .drop(columns = [
                        "const", "beta", "fx_rtn", "fx_ticker", 
                        "pred", "signal", "ticker"]))

            df_add = (pd
                    .concat([df_full, df_train, df_expanding])
                    .assign(
                        signal_name = key,
                        fx_ticker   = fx_ticker,
                        etf_ticker  = etf_ticker)
                    .merge(right = df_tmp_fx, how = "inner", on = ["date", "fx_ticker"]))

            df_lists.append(df_add)
            
        df_prep = (pd
                .concat(df_lists)
                .assign(
                    signal_rtn = lambda x: -np.sign(x.lag_resid) * x.fx_rtn,
                    name       = lambda x: x.regression + " " + x.fx_ticker + " " + etf_ticker))
        
        df_out = self._vol_hedge(df_prep, self.vol_target, self.vol_window, self.threshold)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_optimized_trend_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Geting Optimized Trend Backtest")
        
        sig_path = os.path.join(self.data_path,"OptimizedSignal", "OptimizedTrendSignal.parquet")
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "OptimizedTrendBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_prep = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .merge(right = df_fx_rtn, how = "inner", on = ["date", "fx_ticker"])
                .assign(
                    signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fx_rtn,
                    name       = lambda x: x.fx_ticker + " " + x.optimization))
        
        df_out = self._vol_hedge(df_prep, self.vol_target, self.vol_window, self.threshold)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_optimized_residual_backtest(self, verbose: bool = True) -> None: 
        
        signal_path = os.path.join(self.data_path, "OptimizedSignal", "OptimizedOLSResidual.parquet")
        out_path    = os.path.join(self.data_path, "Backtests", "OptimizedResidualBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_signal = (pd
                     .read_parquet(path = signal_path, engine = "pyarrow")
                     .assign(
                         signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fx_rtn,
                         name       = lambda x: x.fx_ticker + "-" + x.optimization)
                     .groupby("optimization")
                     .apply(self._vol_hedge, self.vol_target, self.vol_window, self.threshold)
                     .reset_index()
                     .drop(columns = ["level_1"]))
        
        if verbose: print("Saving data\n")
        df_signal.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_factor_trend_forecasted_backtest(
            self, 
            verbose   : bool = True,
            slice_year: int  = 2018) -> None: 
        
        if verbose: 
            print("Getting Factor Trend Forecasted Backtest")
        
        sig_path = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "FactorForecastedTrend.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"]))
        
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .loc[lambda x: x.calc == "adjusted"]
                .drop(columns = ["signal", "spread", "value", "SPY", "calc"]))
        
        model_path = os.path.join(self.data_path, "Signals", "ForecastedFactors.pkl")
        with open(model_path, "rb") as f: models = pickle.load(f)
        
        keys     = list(models.keys())
        df_names = (pd
                .DataFrame(data = keys, columns = ["name"])
                .assign(
                    str_split = lambda x: x.name.str.split("-"),
                    fx_ticker  = lambda x: x.str_split.str[0],
                    vol_target = lambda x: x.str_split.str[1],
                    etf_ticker = lambda x: x.str_split.str[2],
                    calc       = lambda x: x.str_split.str[3],
                    field      = lambda x: x.str_split.str[4])
                .loc[lambda x: x.vol_target == "perfect"]
                .loc[lambda x: x.calc == "adjusted"]
                .drop(columns = ["vol_target", "calc", "str_split"]))
        
        names    = df_names.name.drop_duplicates().sort_values().to_list()
        df_lists = []
        
        if verbose: iterable = tqdm(names)
        else      : iterable = names
        
        for name in iterable: 
            
            tmp_model = models[name]
            tmp_dict   = (df_names
                    .loc[lambda x: x.name == name]
                    .iloc[0]
                    .to_dict())
            
            fx_ticker  = tmp_dict["fx_ticker"]
            etf_ticker = tmp_dict["etf_ticker"]
            field      = tmp_dict["field"]
            
            df_tmp_fx_rtn = (df_fx_rtn
                    .loc[lambda x: x.security == fx_ticker]
                    .rename(columns = {"security": "fx_ticker"}))
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.etf_ticker == etf_ticker]
                    .loc[lambda x: x.field == field]
                    .drop(columns = ["source"]))
            
            # in-sample
            alpha, beta = tmp_model["full"]["const"], tmp_model["full"]["lag_signal"]
            df_is = (df_tmp_signal
                            .assign(
                                forecasted   = lambda x: (x.lag_signal * beta) + alpha,
                                optimization = "full_sample",
                                sample_group = "in_sample"))
            
            #in-sample/out-sample
            alpha, beta   = tmp_model["train"]["const"], tmp_model["train"]["lag_signal"]
            df_train_test = (df_tmp_signal
                    .assign(
                        forecasted   = lambda x: (x.lag_signal * beta) + alpha,
                        optimization = "train_test",
                        year         = lambda x: x.date.dt.year,
                        sample_group = lambda x: np.where(x.year <= slice_year, "in_sample", "out_sample"))
                    .drop(columns = ["year"]))
            
            # expanding
            df_expanding = (tmp_model
                    ["expanding"]
                    .shift()
                    .rename(columns = {"lag_signal": "beta"})
                    .merge(right = df_tmp_signal, how = "inner", on = ["date"])
                    .assign(
                        optimization = "expanding",
                        sample_group = "out_sample",
                        forecasted   = lambda x: (x.beta * x.lag_signal) + x.const)
                    .drop(columns = ["const", "beta"]))
            
            df_add = (pd
                    .concat([df_is, df_train_test, df_expanding])
                    .merge(right = df_tmp_fx_rtn, how = "inner", on = ["date"])
                    .assign(
                        signal_rtn = lambda x: np.sign(x.forecasted) * x.fx_rtn,
                        name       = lambda x: x.fx_ticker + "-" + x.optimization)
                    .pipe(lambda x: self._vol_hedge(x)))
    
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_factor_trend_residual_backtest(self, verbose: bool = True, slice_year: int = 2018) -> None: 
        
        if verbose: 
            print("Getting Factor Trend Residual Backtest")
        
        sig_path = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "FactorResidTrend.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_raw_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"]))
        
        df_perf_fx_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"]))
    
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .loc[lambda x: x.calc == "adjusted"]
                .drop(columns = ["lag_signal", "spread", "value", "SPY", "calc"]))
        
        model_path = os.path.join(self.data_path, "Signals", "ResidFactors.pkl")
        with open(model_path, "rb") as f: models = pickle.load(f)
        
        keys     = list(models.keys())
        df_names = (pd
                .DataFrame(data = keys, columns = ["name"])
                .assign(
                    str_split = lambda x: x.name.str.split("-"),
                    fx_ticker  = lambda x: x.str_split.str[0],
                    vol_target = lambda x: x.str_split.str[1],
                    etf_ticker = lambda x: x.str_split.str[2],
                    calc       = lambda x: x.str_split.str[3],
                    field      = lambda x: x.str_split.str[4])
                .loc[lambda x: x.vol_target == "perfect"]
                .loc[lambda x: x.calc == "adjusted"]
                .drop(columns = ["vol_target", "calc", "str_split"]))
        
        names    = df_names.name.drop_duplicates().sort_values().to_list()
        df_lists = []
        
        if verbose: iterable = tqdm(names)
        else      : iterable = names
        
        for name in iterable: 
            
            tmp_model = models[name]
            tmp_dict   = (df_names
                    .loc[lambda x: x.name == name]
                    .iloc[0]
                    .to_dict())
            
            fx_ticker  = tmp_dict["fx_ticker"]
            etf_ticker = tmp_dict["etf_ticker"]
            field      = tmp_dict["field"]
            
            df_tmp_perf_fx = (df_perf_fx_rtn
                    .loc[lambda x: x.security == fx_ticker]
                    .rename(columns = {"security": "fx_ticker"}))
            
            df_tmp_raw_fx = (df_raw_fx_rtn
                    .loc[lambda x: x.security == fx_ticker]
                    .rename(columns = {"security": "fx_ticker"}))
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.etf_ticker == etf_ticker]
                    .loc[lambda x: x.field == field]
                    .drop(columns = ["source", "field", "etf_ticker"]))
            
            # in-sample
            alpha, beta = tmp_model["full"]["const"], tmp_model["full"]["signal"]
            df_is       = (df_tmp_signal
                    .merge(right = df_tmp_perf_fx, how = "inner", on = ["date"])
                    .assign(
                        sample_group = "in_sample",
                        optimization = "full_sample",
                        forecasted   = lambda x: (x.signal * beta) + alpha,
                        resid        = lambda x: x.fx_rtn - x.forecasted,
                        lag_resid    = lambda x: x.resid.shift())
                    .dropna()
                    .drop(columns = ["fx_rtn"]))
            
            #in-sample/out-sample
            alpha, beta   = tmp_model["train"]["const"], tmp_model["train"]["signal"]
            df_train_test = (df_tmp_signal
                    .merge(right = df_tmp_perf_fx, how = "inner", on = ["date"])
                    .assign(
                        year         = lambda x: pd.to_datetime(x.date).dt.year,
                        sample_group = lambda x: np.where(x.year <= slice_year, "in_sample", "out_sample"),
                        optimization = "train_test",
                        forecasted   = lambda x: (x.signal * beta) + alpha,
                        resid        = lambda x: x.fx_rtn - x.forecasted,
                        lag_resid    = lambda x: x.resid.shift())
                    .drop(columns = ["fx_rtn", "year"]))
            
            # expanding
            df_expanding = (tmp_model
                    ["expanding"]
                    .shift()
                    .rename(columns = {"signal": "beta"})
                    .merge(right = df_tmp_signal, how = "inner", on = ["date"])
                    .merge(right = df_tmp_perf_fx, how = "inner", on = ["date"])
                    .assign(
                        optimization = "expanding",
                        sample_group = "out_sample",
                        forecasted   = lambda x: (x.signal * beta) + alpha,
                        resid        = lambda x: x.fx_rtn - x.forecasted,
                        lag_resid    = lambda x: x.resid.shift())
                    .drop(columns = ["fx_rtn", "const", "beta"]))
            
            df_add = (pd
                    .concat([df_is, df_train_test, df_expanding])
                    .merge(right = df_tmp_raw_fx, how = "inner", on = ["date", "fx_ticker"])
                    .assign(
                        signal_rtn = lambda x: -np.sign(x.lag_resid) * x.fx_rtn,
                        name       = lambda x: x.fx_ticker + "-" + x.optimization)
                    .pipe(lambda x: self._vol_hedge(x)))
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_ml_factor_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting ML Backtests")
        
        sig_path = os.path.join(self.data_path, "Signals", "MLFactorForecast.parquet")
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "MLFactorBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "none"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_out = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["index"])
                .merge(right = df_rtn, how = "inner", on = ["date", "fx_ticker"])
                .assign(
                    signal_rtn = lambda x: np.sign(x.forecasted) * x.fx_rtn,
                    name       = lambda x: x.fx_ticker + "-" + x.model)
                .pipe(lambda x: self._vol_hedge(x)))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_single_stock_forecasted_backtest(self, verbose: bool = True) -> None:
        
        if verbose: 
            print("Getting Single Stock Forecasted Backtest")
        
        for_path = os.path.join(self.data_path, "Signals", "SingleNameForecastedFittedValues.parquet")
        rtn_path = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "SingleNameOLSForecasted.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_forecasted = pd.read_parquet(path = for_path, engine = "pyarrow")
        df_rtn        = (pd
                         .read_parquet(path = rtn_path, engine = "pyarrow")
                         .loc[lambda x: x.vol_target == "none"]
                         .rename(columns = {"security": "fx_ticker"})
                         .drop(columns = ["vol_target"]))
        
        df_out = (df_rtn
                .merge(right = df_forecasted, how = "inner", on = ["date", "fx_ticker"])
                .assign(
                    signal_rtn = lambda x: np.sign(x.pred) * x.fx_rtn,
                    name       = lambda x: x.fx_ticker + "-" + x.regression)
                .pipe(lambda x: self._vol_hedge(x, self.vol_target, self.vol_window, self.threshold)))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    backtesting = Backtesting()
    #backtesting.get_spread_backtest()
    #backtesting.get_spread_trend_backtest()
    #backtesting.get_forecasted_ols_backtest() 
    #backtesting.get_residual_ols_backtest()
    #backtesting.get_optimized_trend_backtest()
    #backtesting.get_optimized_residual_backtest()
    #backtesting.get_factor_trend_forecasted_backtest(verbose = True)
    #backtesting.get_factor_trend_residual_backtest(verbose = True)
    #backtesting.get_ml_factor_backtest()
    #backtesting.get_single_stock_forecasted_backtest()
    
if __name__ == "__main__": main()