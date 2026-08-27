# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:44:26 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
from   sklearn.decomposition import PCA

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

class SignalGenerator:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.sig_path  = os.path.join(self.data_path, "Signals")
        
        if not os.path.exists(self.sig_path):
            os.makedirs(self.sig_path)
        
        self.bad_tickers = ["USDPHPCR Curncy"]
            
    def _lag_signal(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df
                  .sort_index()
                  .assign(lag_signal = lambda x: x.signal.shift()))
        
        return df_out
            
    def generate_etf_spread_rtn(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Returns Differential Spread")
        
        in_path  = os.path.join(self.data_path , "PrepData", "ETFReturns.parquet")
        out_path = os.path.join(self.sig_path, "ReturnSpreadSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .pivot(index = ["date"], columns = "ticker", values = "rtn")
                .reset_index()
                .melt(id_vars = ["date", "SPY"])
                .dropna()
                .assign(spread = lambda x: x.value - x.SPY))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def generate_etf_spread_trend(self, window: int = 30, verbose: bool = True) -> None:
        
        if verbose: print("Getting Returns Differential Trend")
        
        in_path  = os.path.join(self.sig_path, "ReturnSpreadSignal.parquet")
        out_path = os.path.join(self.sig_path, "ReturnTrendSpreadSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .pivot(index = "date", columns = ["ticker"], values = "spread")
                .apply(lambda x: 
                       (x - x.ewm(span = window, adjust = False).mean()) / 
                       x.ewm(span = window, adjust = False).std())
                .reset_index()
                .melt(id_vars = "date")
                .rename(columns = {"value": "signal"})
                .dropna()
                .set_index("date")
                .groupby("ticker")
                .apply(self._lag_signal)
                .reset_index()
                .assign(
                    window      = window,
                    signal_name = "ReturnSpreadTrend"))

        if verbose:  print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_ols_model(
            self, 
            df        : pd.DataFrame,
            exog_name : str,
            slice_year: int = 2018, 
            min_nobs  : int = 30) -> dict: 
        
        df_input = (df
                .set_index("date")
                [["fx_rtn", exog_name]]
                .dropna())
        
        # full sample in-sample
        fs_model = (sm
                .OLS(
                    endog = df_input.fx_rtn,
                    exog  = sm.add_constant(df_input[exog_name]))
                .fit())
        
        # train/test split model
        df_train = (df
                .assign(year = lambda x: x.date.dt.year)
                .loc[lambda x: x.year <= slice_year]
                .set_index("date")
                [["fx_rtn", exog_name]]
                .dropna())
        
        train_test_model = (sm
                .OLS(
                    endog = df_train.fx_rtn,
                    exog  = sm.add_constant(df_train[exog_name]))
                .fit())
        
        expanding_model = (RollingOLS(
            endog     = df_input.fx_rtn, 
            exog      = sm.add_constant(df_input[exog_name]),
            min_nobs  = min_nobs,
            expanding = True)
            .fit())
        
        out_dict = {
            "full"     : fs_model,
            "train"    : train_test_model,
            "expanding": expanding_model}
        
        return out_dict
    
    def _get_multi_ols_forecast(
            self, 
            df        : pd.DataFrame,
            exog_list : list,
            slice_year: int = 2018, 
            min_nobs  : int = 30) -> dict: 
        
        df_input = (
            df
            .set_index("date")
            [["fx_rtn"] + exog_list])
    
        regressors = len(exog_list) + 1
    
        if min_nobs < regressors:
            min_nobs = regressors
    
        X = sm.add_constant(df_input[exog_list])
        y = df_input["fx_rtn"]
    
        fs_model = (sm
                    .OLS(
                        endog=y,
                        exog=X).
                        fit())
    
        full_forecast = (fs_model
                         .predict(X)
                         .to_frame(name = "pred")
                         .assign(
                             regression   = "full_sample",
                             sample_group = "full_sample"))
    
        df_train = (
            df
            .assign(year=lambda x: x.date.dt.year)
            .loc[lambda x: x.year <= slice_year]
            .set_index("date")
            [["fx_rtn"] + exog_list]
            .bfill()
            .ffill())
    
        X_train = sm.add_constant(df_train[exog_list])
        y_train = df_train["fx_rtn"]
    
        train_model = (sm
                       .OLS(
                            endog=y_train,
                            exog=X_train)
                       .fit())
    
        train_forecast = (train_model
                .predict(X)
                .to_frame(name = "pred")
                .reset_index()
                .assign(
                    regression   = "train_test",
                    year         = lambda x: x.date.dt.year,
                    sample_group = lambda x: np.where(x.year <= slice_year, "in_sample", "out_sample"))
                .drop(columns = ["year"])
                .set_index("date"))
    
        expanding_model = (RollingOLS(
            endog     = y,
            exog      = X,
            min_nobs  = min_nobs,
            expanding = True)
            .fit())
    
        expanding_params = expanding_model.params
    
        expanding_forecast = (
            X
            .mul(expanding_params)
            .sum(axis=1)
            .to_frame(name = "pred")
            .assign(
                regression   = "expanding",
                sample_group = "out_sample"))
    
    
        df_out = (pd
                .concat([full_forecast, train_forecast, expanding_forecast]))
        
        return df_out
        
    def generate_ols_spread_trend_forecast(
            self, 
            slice_year: int  = 2018,
            verbose   : bool = True) -> None: 
        
        if verbose: print("Getting Predicted Values")
        
        out_path = os.path.join(self.data_path, "Signals", "ForecastedOLS.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .drop(columns = ["name", "etf_benchmark", "CurveFile"])
                .loc[lambda x: x.Active == True]
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        sig_path  = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["signal", "window", "signal_name"])
                .rename(columns = {"ticker": "etf_ticker"}))
        
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_combined = (df_fx_rtn
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_signal, how = "inner", on = ["date", "etf_ticker"])
                .assign(group_var = lambda x: x.fx_ticker + "-" + x.etf_ticker))
        
        group_vars = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_models = {}
        
        if verbose: iterable = tqdm(group_vars)
        else      : iterable = group_vars
        
        for group_var in iterable: 
            
            df_input = (df_combined
                    .loc[lambda x: x.group_var == group_var])
            
            models = self._get_ols_model(df_input, "lag_signal")
            out_models[group_var] = models
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_models, f)
        
    def generate_ols_spread_trend_resid(
            self, 
            slice_year: int  = 2018,
            verbose   : bool = True) -> None: 
        
        if verbose: print("Getting Residual Values")
        
        out_path = os.path.join(self.data_path, "Signals", "ResidOLS.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .drop(columns = ["name", "etf_benchmark", "CurveFile"])
                .loc[lambda x: x.Active == True]
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        sig_path  = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["lag_signal", "window", "signal_name"])
                .rename(columns = {"ticker": "etf_ticker"}))

        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"}))
        
        df_combined = (df_fx_rtn
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_signal, how = "inner", on = ["date", "etf_ticker"])
                .assign(group_var = lambda x: x.fx_ticker + "-" + x.etf_ticker))
        
        group_vars = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_models = {}
        
        for group_var in tqdm(group_vars): 
            
            df_input = (df_combined
                    .loc[lambda x: x.group_var == group_var])
            
            models = self._get_ols_model(df_input, "signal")
            out_models[group_var] = models
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_models, f)
            
    def _get_factor_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df
                .sort_index()
                .assign(
                    signal = lambda x: 
                        (x.spread - x.spread.ewm(span = window, adjust = False).mean()) / 
                        x.spread.ewm(span = window, adjust = False).std(),
                    lag_signal = lambda x: x.signal.shift())
                .reset_index())
            
        return df_out
            
    def generate_factor_spread_trend(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Factor Signal")
        
        prep_path = os.path.join(self.data_path, "PrepData", "FactorData.parquet")
        out_path  = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = prep_path, engine = "pyarrow")
                .set_index("date")
                .assign(tmp = lambda x: x.calc + "-" + x.etf_ticker + "-" + x.field + "-" + x.source)
                #.loc[lambda x: x.tmp == x.tmp.min()]
                .groupby("tmp")
                #.progress_apply(lambda group: self._get_factor_zscore(group, window))
                .apply(self._get_factor_zscore, window)
                .reset_index(drop = True))
            
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _fast_ols_resid(
            self,
            df: pd.DataFrame,
            exog_name: str,
            slice_year: int = 2018,
            min_nobs: int = 30,
        ) -> dict:
    
        # --------------------------------------------------
        # Prepare data once
        # --------------------------------------------------
    
        df_input = (
            df[["date", "fx_rtn", exog_name]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )
    
        y = df_input["fx_rtn"].to_numpy(dtype=float)
        x = df_input[exog_name].to_numpy(dtype=float)
    
        # ==================================================
        # FULL SAMPLE
        # ==================================================
    
        n = len(y)
    
        x_mean = x.mean()
        y_mean = y.mean()
    
        x_d = x - x_mean
        y_d = y - y_mean
    
        sxx = np.sum(x_d ** 2)
        sxy = np.sum(x_d * y_d)
    
        beta_full = sxy / sxx
        alpha_full = y_mean - beta_full * x_mean
    
        # ==================================================
        # TRAIN
        # ==================================================
    
        train_mask = (
            df_input["date"].dt.year.to_numpy() <= slice_year
        )
    
        x_train = x[train_mask]
        y_train = y[train_mask]
    
        x_mean = x_train.mean()
        y_mean = y_train.mean()
    
        x_d = x_train - x_mean
        y_d = y_train - y_mean
    
        sxx = np.sum(x_d ** 2)
        sxy = np.sum(x_d * y_d)
    
        beta_train = sxy / sxx
        alpha_train = y_mean - beta_train * x_mean
    
        # ==================================================
        # EXPANDING
        #
        # Calculate OLS parameters at every date using
        # cumulative sufficient statistics.
        # ==================================================
    
        n_obs = np.arange(1, len(x) + 1)
    
        sx = np.cumsum(x)
        sy = np.cumsum(y)
        sxx_raw = np.cumsum(x ** 2)
        syy_raw = np.cumsum(y ** 2)
        sxy_raw = np.cumsum(x * y)
    
        # Centered sums
        Sxx = sxx_raw - sx ** 2 / n_obs
        Sxy = sxy_raw - sx * sy / n_obs
    
        beta_expanding = Sxy / Sxx
        alpha_expanding = (
            sy / n_obs
            - beta_expanding * sx / n_obs
        )
    
        # Don't use estimates before min_nobs
        beta_expanding[:min_nobs - 1] = np.nan
        alpha_expanding[:min_nobs - 1] = np.nan
    
        expanding_params = pd.DataFrame(
            {
                "const": alpha_expanding,
                exog_name: beta_expanding,
            },
            index=pd.DatetimeIndex(df_input["date"]),
        )
    
        # ==================================================
        # Return
        # ==================================================
    
        out_dict =  {
            "full": {
                "const": alpha_full,
                exog_name: beta_full,
            },
    
            "train": {
                "const": alpha_train,
                exog_name: beta_train,
            },
    
            "expanding": expanding_params,
        }
        
        return out_dict
        
    def generate_ols_factor_forecast(self, verbose: bool = True) -> None: 
        
        signal_path = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")        
        tick_path   = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.data_path, "Signals", "ForecastedFactors.pkl")
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            [["etf_ticker", "fx_ticker"]]
            .dropna()
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        df_signal = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["signal", "value", "SPY", "spread"])
                .dropna())
        
        df_combined = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .rename(columns = {"security": "fx_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_signal, how = "inner", on = ["date", "etf_ticker"])
                .dropna()
                .assign(group_var = lambda x: 
                        x.fx_ticker  + "-" + 
                        x.vol_target + "-" + 
                        x.etf_ticker + "-" + 
                        x.calc       + "-" + 
                        x.field      + "-" + 
                        x.source     + "-"))
            
        group_vars = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_models = {}
        
        if verbose: iterable = tqdm(group_vars)
        else      : iterable = group_vars

        for group_var in iterable: 
            
            try:
            
                df_input = (df_combined
                        .loc[lambda x: x.group_var == group_var])
                
                models = self._fast_ols_resid(df_input, "lag_signal")
                out_models[group_var] = models
                
            except: 
                if verbose: print("Failed at", group_var)

        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_models, f)
            
    def generate_ols_resid_factor_forecast(self, verbose: bool = True) -> None:
        
        signal_path = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")        
        tick_path   = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.data_path, "Signals", "ResidFactors.pkl")
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            [["etf_ticker", "fx_ticker"]]
            .dropna()
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        df_signal = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["spread", "lag_signal", "SPY", "value"]))
        
        df_combined = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .rename(columns = {"security": "fx_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_signal, how = "inner", on = ["date", "etf_ticker"])
                .dropna()
                .assign(group_var = lambda x: 
                        x.fx_ticker  + "-" + 
                        x.vol_target + "-" + 
                        x.etf_ticker + "-" + 
                        x.calc       + "-" + 
                        x.field      + "-" + 
                        x.source     + "-"))
            
        group_vars = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_models = {}
        
        if verbose: iterable = tqdm(group_vars)
        else      : iterable = group_vars

        for group_var in iterable: 
            
            try:
            
                df_input = (df_combined
                        .loc[lambda x: x.group_var == group_var])
                
                models = self._fast_ols_resid(df_input, "signal")
                out_models[group_var] = models
                
            except: 
                if verbose: print("Failed at", group_var)
                
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_models, f)
            
    def get_ml_model_forecast(self, slice_year: int = 2018, verbose: bool = True) -> pd.DataFrame:
        
        if verbose: print("Getting ML Forecasted Data")
    
        model_path = os.path.join( self.data_path, "MLModels", "Models.pkl")
        sig_path   = os.path.join(self.data_path, "PrepData", "MLFactors.parquet")
        out_path   = os.path.join(self.data_path, "Signals", "MLFactorForecast.parquet")
    
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
    
        with open(model_path, "rb") as f:
            models = pickle.load(f)
    
        df_signal = pd.read_parquet(path = sig_path, engine = "pyarrow")
        df_lists  = []
    
        for fx_ticker, tmp_models in models.items():
    
            df_tmp_signal = (df_signal
                .loc[lambda x: x.fx_ticker == fx_ticker]
                .drop(columns=["fx_ticker", "etf_ticker"])
                .pivot(index = ["date", "fx_rtn"], columns = "variable", values = "value")
                .reset_index()
                .set_index("date"))
    
            # Keep sample_group for the final output
            X_cols = [
                col
                for col in df_tmp_signal.columns
                if col not in ["fx_rtn", "sample_group"]]
    
            X = df_tmp_signal[X_cols]
            y = df_tmp_signal["fx_rtn"]
    
            for model_type, model in tmp_models.items():
    
                y_hat = model.predict(X)
    
                df_add = (df_tmp_signal
                        .reset_index()
                        [["date"]]
                        .assign(
                            forecasted = y_hat, 
                            fx_ticker  = fx_ticker,
                            model      = model_type)
                        .reset_index())
                
                df_lists.append(df_add)
    
        df_forecasts = (pd
                        .concat(df_lists, axis=0, ignore_index=True)
                        .set_index("date")
                        .sort_index())
    
        df_forecasts.to_parquet(
            out_path,
            engine="pyarrow")
    
        if verbose: print("Saving data\n")
        df_forecasts.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_single_name_stock_trend(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Single-Name Stock Trend Data")
        
        data_path = os.path.join(self.data_path, "PrepData", "SingleNameData.parquet")
        out_path  = os.path.join(self.data_path, "Signals", "SingleNameStockSpreadTrend.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_signal = (pd
                .read_parquet(path = data_path, engine = "pyarrow")
                .dropna()
                .groupby(["date", "exch", "stock_ticker", "etf_ticker", "provider", "spy_rtn"])
                .agg("mean")
                .reset_index()
                .assign(spread = lambda x: x.rtn - x.spy_rtn)
                .drop(columns = ["spy_rtn", "provider", "rtn"])
                .set_index("date")
                .groupby(["exch", "stock_ticker", "etf_ticker"])
                .apply(lambda x: 
                       (x.spread - x.spread.ewm(span = window, adjust = False).mean()) / 
                       x.spread.ewm(span = window, adjust = False).std())
                .reset_index()
                .rename(columns = {"spread": "signal"}))
            
        df_out = (df_signal
                .set_index("date")
                .groupby(["exch", "stock_ticker", "etf_ticker"])
                .apply(lambda x: x.signal.shift())
                .reset_index()
                .rename(columns = {"signal": "lag_signal"})
                .merge(
                    right = df_signal, 
                    how   = "inner", 
                    on    = ["exch", "stock_ticker", "etf_ticker", "date"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_single_stock_forecast(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Single Stock Forecast Models")
        
        signal_path = os.path.join(self.data_path, "Signals", "SingleNameStockSpreadTrend.parquet")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.data_path, "Signals", "SingleNameForecastedFittedValues.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_ticker = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .loc[lambda x: x.Active == True]
                [["ticker", "etf_ticker"]]
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
                .loc[lambda x: ~x.fx_ticker.isin(self.bad_tickers)]
                .sort_values("fx_ticker"))
        
        df_signal = (pd
                .read_parquet(path = signal_path, engine = "pyarrow")
                .drop(columns = ["signal"])
                .dropna()
                .merge(right = df_ticker, how = "inner", on = ["etf_ticker"]))
        
        df_raw_combined = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"})
                .merge(right = df_signal, how = "inner", on = ["date", "fx_ticker"]))
        
        df_selector = (df_raw_combined
                [["date", "fx_ticker", "etf_ticker", "stock_ticker"]]
                .groupby(["fx_ticker", "etf_ticker", "stock_ticker"])
                ["date"]
                .agg(["min", "max", "count"])
                .rename(columns = {"min": "start_date"})
                .reset_index()
                .assign(year = lambda x: x.start_date.dt.year)
                .loc[lambda x: x.year <= 2015]
                .drop(columns = ["max", "count", "year", "start_date"]))
        
        df_combined = (df_raw_combined
                .merge(
                    right = df_selector, 
                    how   = "inner", 
                    on    = ["fx_ticker", "etf_ticker", "stock_ticker"]))
        
        df_lists = []
        
        for i, row in df_ticker.iterrows():
            
            row_dict = row.to_dict()
            
            print("Working on {}".format(row_dict["fx_ticker"]))
            
            df_tmp = (df_combined
                    .loc[lambda x: x.fx_ticker == row_dict["fx_ticker"]]
                    .loc[lambda x: x.etf_ticker == row_dict["etf_ticker"]])
            
            df_replacer = (df_tmp
                .assign(year = lambda x: x.date.dt.year)
                .loc[lambda x: x.year <= 2018]
                [["stock_ticker", "lag_signal"]]
                .groupby("stock_ticker")
                .agg("median")
                .rename(columns = {"lag_signal": "replacer"}))
            
            df_input = (df_tmp
                .dropna()
                [["date", "fx_rtn", "stock_ticker", "lag_signal"]]
                .groupby(["date", "fx_rtn", "stock_ticker"])
                .agg("mean")
                .reset_index()
                .pivot(index = ["date", "fx_rtn"], columns = "stock_ticker", values = "lag_signal")
                .reset_index()
                .melt(id_vars = ["date", "fx_rtn"])
                .sort_values("value")
                .merge(right = df_replacer, how = "inner", on = ["stock_ticker"])
                .assign(repl_val = lambda x: np.where(x.value != x.value, x.replacer, x.value))
                .pivot(index = ["date", "fx_rtn"], columns = "stock_ticker", values = "repl_val")
                .reset_index())
            
            all_cols  = df_input.columns.to_list()
            exog_cols = list(set(all_cols) - set(["fx_rtn", "date"]))
            
            df_add = (self
                           ._get_multi_ols_forecast(df_input, exog_cols)
                           .assign(
                               fx_ticker  = row_dict["fx_ticker"],
                               etf_ticker = row_dict["etf_ticker"]))
            
            df_lists.append(df_add)
        
        df_out = pd.concat(df_lists)
        if verbose: 
            print("Saving data\n")
            df_out.to_parquet(path = out_path, engine = "pyarrow")
    
def main() -> None: 
            
    signal_generator = SignalGenerator()
    #signal_generator.generate_etf_spread_rtn()
    #signal_generator.generate_etf_spread_trend()
    #signal_generator.generate_ols_spread_trend_forecast(verbose = True)
    #signal_generator.generate_ols_spread_trend_resid()
    #signal_generator.generate_factor_spread_trend()
    #signal_generator.generate_ols_factor_forecast(verbose = True)
    #signal_generator.generate_ols_resid_factor_forecast()
    #signal_generator.get_ml_model_forecast()
    signal_generator.get_single_name_stock_trend()
    signal_generator.get_single_stock_forecast()

if __name__ == "__main__": main()