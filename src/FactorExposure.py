# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:41:02 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from scipy.stats import t


class FactorExposure:
    
    def __init__(self) -> None: 
        
        self.src_path    = os.getcwd()
        self.repo_path   = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
        self.factor_path = os.path.join(self.data_path, "FactorAnalysis")
        
        if not os.path.exists(self.factor_path):
            os.makedirs(self.factor_path)
            
    def _ols_models(self, df: pd.DataFrame, exog_var: str) -> dict:
        
        groups = df.group_var.drop_duplicates().sort_values().to_list()
        models = {}
        
        for group in tqdm(groups): 
        
            df_tmp = (df
                .loc[lambda x: x.group_var == group]
                .set_index("date")
                .dropna())
        
            model = (sm
                .OLS(
                    endog = df_tmp.fx_rtn,
                    exog  = sm.add_constant(df_tmp[exog_var]))
                .fit())
        
            models[group] = model
            
        return models
            
    def generate_etf_rtn_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting ETF Returns Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsOLSModels.pkl")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "PrepData", "ETFReturns.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        df_etf = (pd
                .read_parquet(path = etf_path, engine = "pyarrow")
                .rename(columns = {
                    "ticker": "etf_ticker",
                    "rtn"   : "etf_rtn"}))

        df_fx_rtn = (pd
            .read_parquet(path = fx_path, engine = "pyarrow")
            .rename(columns = {"security": "fx_ticker"}))
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize())
            .drop(columns = [
                "name", "index_ticker", "CurveFile", "fx_name",
                "country", "shorthand_name"]))
        
        df_combined = (df_fx_rtn
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_etf   , how = "inner", on = ["etf_ticker", "date"])
                .assign(group_var = lambda x: x.fx_ticker + "-" + x.vol_target + "-" + x.return_type + "-" + x.etf_ticker)
                .dropna())
        
        models = self._ols_models(df_combined, "etf_rtn")
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)

    def get_etf_return_differential_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Sovereign Returns Spread Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsDifferentialOLSModels.pkl")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None        
        etf_path  = os.path.join(self.data_path, "Signals", "ReturnSpreadSignal.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = [
                "Active", "FACTOR_ACTIVE", "name", "index_ticker", 
                "CurveFile"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize()))
        
        df_etf_spread = (pd
            .read_parquet(path = etf_path, engine = "pyarrow")
            .drop(columns = ["value", "SPY"])
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
            .read_parquet(path = fx_path, engine = "pyarrow")
            .rename(columns = {"security": "fx_ticker"}))
        
        df_combined = (df_fx_rtn
            .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
            .merge(right = df_etf_spread, how = "inner", on = ["date", "etf_ticker"])
            .assign(group_var = lambda x: x.fx_ticker + "-" + x.vol_target + "-" + x.return_type + "-" + x.etf_ticker)
            .dropna())
        
        models = self._ols_models(df_combined, "spread")
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def get_etf_return_trend_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Sovereign Returns Spread Trend Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsDifferentialTrendOLSModels.pkl")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None        
        
        etf_path  = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = [
                "Active", "FACTOR_ACTIVE", "name", "index_ticker", 
                "CurveFile"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize()))
        
        df_etf_spread = (pd
            .read_parquet(path = etf_path, engine = "pyarrow")
            .drop(columns = ["window", "signal_name"])
            .rename(columns = {
                "signal"    : "no_lag",
                "lag_signal": "with_lag"})
            .melt(id_vars = ["date", "ticker"], var_name = "signal_lag", value_name = "signal")
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
            .read_parquet(path = fx_path, engine = "pyarrow")
            .rename(columns = {"security": "fx_ticker"}))
        
        df_combined = (df_fx_rtn
            .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
            .merge(right = df_etf_spread, how = "inner", on = ["date", "etf_ticker"])
            .assign(
                group_var = lambda x: 
                    x.etf_ticker + "-" + 
                    x.fx_ticker + "-" + 
                    x.vol_target + "-" + 
                    x.signal_lag)
            .dropna())
        
        models = self._ols_models(df_combined, "signal")
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)

    
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        return df.sort_index().diff()
    
    def generate_all_factors(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting PCs OLS Models")
        
        out_path = os.path.join(self.factor_path, "PCAOLSModels.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        pc_path   = os.path.join(self.data_path, "PrepData", "YieldCurvePCs.parquet")

        df_pcs = (pd
                .read_parquet(path = pc_path, engine = "pyarrow")
                .set_index("date")
                .groupby(["country", "variable"])
                .apply(self._get_diff)
                .reset_index()
                .pivot(index = ["date", "country"], columns = "variable", values = "value")
                .reset_index())
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize()))
        
        df_spread = (pd
                .read_parquet(path = etf_path, engine = "pyarrow")
                [["date", "ticker", "signal"]]
                .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .rename(columns = {"security": "fx_ticker"}))

        df_lag_fx = (df_fx_rtn
                .dropna()
                .set_index("date")
                .assign(group_var = lambda x: x.fx_ticker + " " + x.vol_target)
                .groupby("group_var")
                .apply(lambda x: x.sort_index().shift())
                .rename(columns = {"fx_rtn": "lag_fx_rtn"})
                .reset_index()
                .drop(columns = ["group_var"])
                .dropna())
        
        df_combined = (df_fx_rtn
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_spread, how = "inner", on = ["date", "etf_ticker"])
                .merge(right = df_lag_fx, how = "inner", on = ["date", "fx_ticker", "vol_target"])
                .merge(right = df_pcs,    how = "inner", on = ["date", "country"])
                .assign(group_var = lambda x: x.fx_ticker + "-" + x.vol_target + "-" + x.etf_ticker))
        
        exog_var = ["signal", "lag_fx_rtn", "PC1", "PC2", "PC3"]
        models   = self._ols_models(df_combined, exog_var)
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def get_factor_etf_models(self, verbose: bool = True) -> None:
    
        if verbose:
            print("Getting Factor ETF Models")
    
        factor_path = os.path.join(
            self.data_path,
            "Signals",
            "FactorSignal.parquet",
        )
    
        rtn_path = os.path.join(
            self.data_path,
            "PrepData",
            "FXVolTargetedReturns.parquet",
        )
    
        tick_path = os.path.join(
            self.data_path,
            "FXTickerGuide.xlsx",
        )
    
        out_path = os.path.join(
            self.data_path,
            "FactorAnalysis",
            "ETFFactors.parquet",
        )
    
        # --------------------------------------------------
        # Already exists
        # --------------------------------------------------
    
        if os.path.exists(out_path):
    
            if verbose:
                print("Already have data\n")
    
            return None
    
        # --------------------------------------------------
        # Ticker guide
        # --------------------------------------------------
    
        df_ticker = (
            pd.read_excel(
                io=tick_path,
                sheet_name="TickerGuide",
                usecols=["etf_ticker", "ticker"],
            )
            .rename(
                columns={
                    "ticker": "fx_ticker",
                }
            )
            .dropna()
            .assign(
                etf_ticker=lambda x:
                    x.etf_ticker.str.split(" ").str[0]
            )
            [["etf_ticker", "fx_ticker"]]
            .drop_duplicates()
        )
    
        # --------------------------------------------------
        # Factor signals
        # --------------------------------------------------
    
        df_signal = (
            pd.read_parquet(
                path=factor_path,
                engine="pyarrow",
                columns=[
                    "date",
                    "calc",
                    "field",
                    "source",
                    "etf_ticker",
                    "signal",
                    "lag_signal",
                ],
            )
            .rename(
                columns={
                    "signal": "no_lag",
                    "lag_signal": "with_lag",
                }
            )
            .melt(
                id_vars=[
                    "date",
                    "calc",
                    "field",
                    "source",
                    "etf_ticker",
                ],
                var_name="signal_lag",
                value_name="signal",
            )
        )
    
        # --------------------------------------------------
        # Returns
        # --------------------------------------------------
    
        df_rtn = (
            pd.read_parquet(
                path=rtn_path,
                engine="pyarrow",
                columns=[
                    "date",
                    "security",
                    "vol_target",
                    "fx_rtn",
                ],
            )
            .rename(
                columns={
                    "security": "fx_ticker",
                }
            )
        )
    
        # --------------------------------------------------
        # Merge
        # --------------------------------------------------
    
        df_prep = (
            df_rtn
            .merge(
                right=df_ticker,
                how="inner",
                on="fx_ticker",
            )
            .merge(
                right=df_signal,
                how="inner",
                on=[
                    "date",
                    "etf_ticker",
                ],
            )
            [
                [
                    "fx_ticker",
                    "vol_target",
                    "etf_ticker",
                    "calc",
                    "field",
                    "source",
                    "signal_lag",
                    "fx_rtn",
                    "signal",
                ]
            ]
            .dropna(
                subset=[
                    "fx_rtn",
                    "signal",
                ]
            )
        )
        
        # --------------------------------------------------
        # Regression groups
        # --------------------------------------------------
    
        group_cols = [
            "fx_ticker",
            "vol_target",
            "etf_ticker",
            "calc",
            "field",
            "source",
            "signal_lag",
        ]
    
        # --------------------------------------------------
        # Sufficient statistics
        #
        # We need:
        #
        # Sxx = Σ(x - x̄)^2
        # Sxy = Σ(x - x̄)(y - ȳ)
        # Syy = Σ(y - ȳ)^2
        #
        # These can be obtained from sums without
        # explicitly fitting an OLS model.
        # --------------------------------------------------
    
        df_prep = df_prep.assign(
            x=df_prep["signal"],
            y=df_prep["fx_rtn"],
            xy=df_prep["signal"] * df_prep["fx_rtn"],
            x2=df_prep["signal"] ** 2,
            y2=df_prep["fx_rtn"] ** 2,
        )
    
        grouped = (
            df_prep
            .groupby(
                group_cols,
                sort=False,
                observed=True,
            )
            [
                [
                    "x",
                    "y",
                    "xy",
                    "x2",
                    "y2",
                ]
            ]
            .agg(
                [
                    "sum",
                ]
            )
        )
    
        n = (
            df_prep
            .groupby(
                group_cols,
                sort=False,
                observed=True,
            )
            .size()
            .rename("n")
        )
    
        # --------------------------------------------------
        # Flatten aggregation columns
        # --------------------------------------------------
    
        grouped.columns = grouped.columns.droplevel(1)
    
        sx = grouped["x"]
        sy = grouped["y"]
        sxy = grouped["xy"]
        sx2 = grouped["x2"]
        sy2 = grouped["y2"]
    
        # --------------------------------------------------
        # Centered sums of squares
        # --------------------------------------------------
    
        sxx = sx2 - (sx ** 2 / n)
    
        sxy = sxy - (sx * sy / n)
    
        syy = sy2 - (sy ** 2 / n)
    
        # --------------------------------------------------
        # OLS beta
        # --------------------------------------------------
    
        beta = sxy / sxx
    
        # --------------------------------------------------
        # Sum squared residuals
        #
        # SSE = Syy - beta * Sxy
        # --------------------------------------------------
    
        sse = syy - beta * sxy
    
        # --------------------------------------------------
        # R-squared
        # --------------------------------------------------
    
        r2 = 1 - (sse / syy)
    
        # --------------------------------------------------
        # Standard error of beta
        # --------------------------------------------------
    
        mse = sse / (n - 2)
    
        se_beta = np.sqrt(
            mse / sxx
        )
    
        # --------------------------------------------------
        # t-statistic
        # --------------------------------------------------
    
        tvalue = beta / se_beta
    
        # --------------------------------------------------
        # p-value
        # --------------------------------------------------
    
        pvalue = (
            2
            * t.sf(
                np.abs(tvalue),
                df=n - 2,
            )
        )
    
        # --------------------------------------------------
        # Results
        # --------------------------------------------------
    
        df_out = (
            pd.DataFrame(
                {
                    "beta": beta,
                    "tvalue": tvalue,
                    "pvalue": pvalue,
                    "r2": r2,
                    "n": n,
                }
            )
            .reset_index()
        )
    
        # --------------------------------------------------
        # Save
        # --------------------------------------------------
    
        if verbose:
            print("Saving data\n")
    
        df_out.to_parquet(
            path=out_path,
            engine="pyarrow",
            index=False,
        )
    
        return None
    
    def get_single_name_factor_exposure(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Factor Models")
        
        sig_path    = os.path.join(self.data_path, "Signals", "SingleNameStockSpreadTrend.parquet")
        rtn_path    = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.data_path, "FactorAnalysis", "SingleNameStocks.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_ticker = (pd
                .read_excel(io = ticker_path)
                .loc[lambda x: x.Active == True]
                [["ticker", "etf_ticker"]]
                .dropna()
                .rename(columns = {"ticker": "fx_ticker"})
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0]))
        
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .drop(columns = ["exch"])
                .melt(id_vars = ["date", "etf_ticker", "stock_ticker"])
                .dropna())
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"]))
        
        df_combined = (df_signal
                .merge(right = df_rtn, how = "inner", on = ["date", "etf_ticker"])
                .assign(name = lambda x: x.fx_ticker + "-" + x.etf_ticker + "-" + x.variable))
        
        names  = df_combined[["name"]].groupby("name").head(1).name.to_list()
        models = {}
        
        if verbose: iterable = tqdm(names)
        else      : iterable = names
        
        for name in iterable: 
            
            df_tmp = df_combined.loc[lambda x: x.name == name]
            
            df_wider = (df_tmp
                    [["date", "fx_rtn", "stock_ticker", "value"]]
                    .groupby(["date", "fx_rtn", "stock_ticker"])
                    .agg("mean")
                    .reset_index()
                    .pivot(index = ["date", "fx_rtn"], columns = "stock_ticker", values = "value")
                    .reset_index()
                    .set_index("date")
                    .dropna())
            
            exog_cols = [col for col in df_wider.columns.to_list() if col != "fx_rtn"]
            model     = (sm
                    .OLS(
                        endog = df_wider.fx_rtn,
                        exog  = sm.add_constant(df_wider[exog_cols]))
                    .fit())
            
            models[name] = model
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
    
    def _get_params(self, model) -> dict: 
        
        rsquared  = model.rsquared
        
        df_param  = model.params.to_frame(name = "coeff_val").reset_index()
        df_pvalue = model.pvalues.to_frame(name = "pvalue").reset_index()
        df_tstat  = model.tvalues.to_frame(name = "tvalue").reset_index()
        
        df_out = (df_param
                .merge(right = df_pvalue, how = "inner", on = ["index"])
                .merge(right = df_tstat, how = "inner", on = ["index"])
                .rename(columns = {"index": "coeff"})
                .loc[lambda x: x.coeff != "const"]
                .melt(id_vars = "coeff")
                .drop(columns = ["coeff"])
                .groupby("variable")
                .agg("mean")
                .T
                .assign(rsquared = rsquared))
        
        return df_out
        
    def compare_etf_stock_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting ETF-Single Name stock comparison")
        
        signal_mapper = {
            "lag_signal": "with_lag",
            "signal"    : "no_lag"}
        
        single_path = os.path.join(self.factor_path, "SingleNameStocks.pkl")
        etf_path    = os.path.join(self.factor_path, "ReturnsDifferentialTrendOLSModels.pkl")
        out_path    = os.path.join(self.factor_path, "SingleStockETFModelComparison.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        with open(single_path, "rb") as f: single_name_models = pickle.load(f)
        with open(etf_path, "rb") as f   : etf_models         = pickle.load(f)
        
        df_etf_model_names = (pd.DataFrame({
            "model_name": etf_models.keys()})
            .assign(fx_vol = lambda x: x.model_name.str.split("-").str[-2])
            .loc[lambda x: x.fx_vol == "perfect"]
            .drop(columns = ["fx_vol"]))
        
        single_model_names = list(single_name_models.keys())[1:]
        df_lists           = []
        
        if verbose: iterable = tqdm(single_model_names)
        else      : iterable = single_name_models
        
        for single_model_name in iterable:
            
            fx_ticker, etf_ticker, signal = single_model_name.split("-")
            
            etf_model_name = (df_etf_model_names
                .assign(
                    signal_name = lambda x: x.model_name.str.split("-").str[-1],
                    fx_name     = lambda x: x.model_name.str.split("-").str[1])
                .loc[lambda x: x.fx_name == fx_ticker]
                .loc[lambda x: x.signal_name == signal_mapper[signal]]
                .model_name
                .item())
            
            tmp_etf_model   = etf_models[etf_model_name]
            tmp_stock_model = single_name_models[single_model_name]
            
            df_etf   = self._get_params(tmp_etf_model)
            df_stock = self._get_params(tmp_stock_model)
            
            df_add = (pd
                    .concat([
                        df_etf.assign(name = "etf"), 
                        df_stock.assign(name = "stock")])
                    .reset_index(drop = True)
                    .assign(
                        etf_ticker = etf_ticker,                        
                        fx_ticker  = fx_ticker,
                        lag        = signal))
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    factor_exposure = FactorExposure()
    #factor_exposure.generate_etf_rtn_models()
    #factor_exposure.get_etf_return_differential_models()
    #factor_exposure.get_etf_return_trend_models()
    #factor_exposure.generate_all_factors()
    #factor_exposure.get_factor_etf_models()
    factor_exposure.get_single_name_factor_exposure()
    factor_exposure.compare_etf_stock_models()
    
if __name__ == "__main__": main()