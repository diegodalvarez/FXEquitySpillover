# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:41:02 2026

@author: Diego
"""

import os
import pickle
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm

class FactorExposure:
    
    def __init__(self) -> None: 
        
        self.src_path    = os.getcwd()
        self.repo_path   = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
        self.factor_path = os.path.join(self.data_path, "FactorExposure")
        
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
            
    def generate_etf_rtn_factor(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting ETF Returns Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsOLSModels.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "PrepData", "ETFVolTargetedReturns.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        df_etf = (pd
            .read_parquet(path = etf_path, engine = "pyarrow")
            .drop(columns = ["PX_LAST", "vol"])
            .melt(id_vars = ["ticker", "date"], value_name = "etf_rtn")
            .dropna()
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx = (pd
            .read_parquet(path = fx_path, engine = "pyarrow")
            .loc[lambda x: x.variable.isin(["px_rtn", "perf_rtn", "lag_rtn"])]
            .rename(columns = {
                "value"   : "fx_rtn",
                "security": "fx_ticker"}))
        
        df_ticker = (pd
            .read_excel(io = tick_path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize()))
        
        df_combined = (df_fx
            .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
            .merge(right = df_etf, how = "inner", on = ["etf_ticker", "variable", "date"])
            .assign(group_var = lambda x: x.etf_ticker + "-" + x.fx_ticker + "-" + x.variable))
        
        models = self._ols_models(df_combined, "etf_rtn")
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def generate_etf_rtn_differential_factor(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Sovereign Returns Spread Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsDifferentialOLSModels.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "PrepData", "ETFVolTargetedReturns.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
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
            .drop(columns = ["PX_LAST", "vol"])
            .melt(id_vars = ["date", "ticker"], value_name = "rtn")
            .pivot(index = ["date", "variable"], columns = "ticker", values = "rtn")
            .reset_index()
            .melt(id_vars = ["date", "variable", "SPY"], value_name = "etf_rtn")
            .dropna()
            .assign(spread = lambda x: x.etf_rtn - x.SPY)
            [["date", "variable", "ticker", "spread"]]
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx = (pd
            .read_parquet(path = fx_path, engine = "pyarrow")
            .loc[lambda x: x.variable.isin(["px_rtn", "perf_rtn", "lag_rtn"])]
            .rename(columns = {
                "value"   : "fx_rtn",
                "security": "fx_ticker"}))
        
        df_combined = (df_fx
            .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
            .merge(right = df_spread, how = "inner", on = ["date", "etf_ticker", "variable"])
            .assign(group_var = lambda x: x.etf_ticker + "-" + x.fx_ticker + "-" + x.variable))
        
        models = self._ols_models(df_combined, "spread")
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def generate_etf_rtn_and_differential_factor(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Sovereign Returns and Returns Spread Models")
        
        out_path = os.path.join(self.factor_path, "ReturnsandReturnsDifferentialOLSModels.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "PrepData", "ETFVolTargetedReturns.parquet")
        fx_path   = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
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
            .drop(columns = ["PX_LAST", "vol"])
            .melt(id_vars = ["date", "ticker"], value_name = "rtn")
            .pivot(index = ["date", "variable"], columns = "ticker", values = "rtn")
            .reset_index()
            .melt(id_vars = ["date", "variable", "SPY"], value_name = "etf_rtn")
            .dropna()
            .assign(spread = lambda x: x.etf_rtn - x.SPY)
            .drop(columns = ["SPY"])
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx = (pd
                 .read_parquet(path = fx_path, engine = "pyarrow")
                 .loc[lambda x: x.variable.isin(["px_rtn", "lag_rtn", "perf_rtn"])]
                 .rename(columns = {
                     "value"   : "fx_rtn",
                     "security": "fx_ticker"}))
        
        df_combined = (df_fx
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_spread, how = "inner", on = ["date", "variable", "etf_ticker"])
                .assign(group_var = lambda x: x.etf_ticker + "-" + x.fx_ticker + "-" + x.variable))
    
        models = self._ols_models(df_combined, ["spread", "etf_rtn"])
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def get_ols_params(self, models: dict) -> pd.DataFrame:
        
        path      = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker = (pd
            .read_excel(io = path, sheet_name = "TickerGuide")
            .rename(columns = {
                "ticker"       : "fx_ticker",
                "etf_benchmark": "index_ticker"})
            .drop(columns = ["Active", "FACTOR_ACTIVE"])
            .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
            .assign(fx_name = lambda x: x.shorthand_name + " " + x.return_type.str.capitalize()))
        
        df_params_list = []
        for key in models.keys():
        
            name               = key.split("-")[1]
            return_type_mapper = (df_ticker
                .set_index("fx_ticker")
                .return_type
                .to_dict())
        
            scaler_name = return_type_mapper[name]
            
            if scaler_name == "carry": scaler = -1
            else                     : scaler = 1
        
            tmp_model = models[key]
            df_param  = tmp_model.params.to_frame(name = "param_val").reset_index()
            df_pvalue = tmp_model.pvalues.to_frame(name = "pvalue").reset_index()
            df_tvalue = tmp_model.tvalues.to_frame(name = "tvalue").reset_index()
        
            df_add = (df_param
                .merge(right = df_pvalue, how = "inner", on = ["index"])
                .merge(right = df_tvalue, how = "inner", on = ["index"])
                .rename(columns = {"index": "param_name"})
                .assign(
                    r2        = tmp_model.rsquared, 
                    param_val = lambda x: scaler * x.param_val,
                    group_var = key,
                    tvalue    = lambda x: scaler * x.tvalue))
        
            df_params_list.append(df_add)
        
        df_params = pd.concat(df_params_list)
        return df_params
    
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        return df.sort_index().diff()
    
    def generate_all_factors(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting PCs OLS Models")
        
        out_path = os.path.join(self.factor_path, "PCAOLSModels.pkl")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        etf_path  = os.path.join(self.data_path, "PrepData", "ETFVolTargetedReturns.parquet")
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
            .drop(columns = ["PX_LAST", "vol"])
            .melt(id_vars = ["date", "ticker"], value_name = "rtn")
            .pivot(index = ["date", "variable"], columns = "ticker", values = "rtn")
            .reset_index()
            .melt(id_vars = ["date", "variable", "SPY"], value_name = "etf_rtn")
            .dropna()
            .assign(spread = lambda x: x.etf_rtn - x.SPY)
            .drop(columns = ["SPY"])
            .rename(columns = {"ticker": "etf_ticker"}))
        
        df_fx = (pd
                 .read_parquet(path = fx_path, engine = "pyarrow")
                 .loc[lambda x: x.variable.isin(["px_rtn", "lag_rtn", "perf_rtn"])]
                 .rename(columns = {
                     "value"   : "fx_rtn",
                     "security": "fx_ticker"}))
        
        df_lag_fx = (df_fx
                .dropna()
                .set_index("date")
                .assign(group_var = lambda x: x.fx_ticker + " " + x.variable)
                .groupby("group_var")
                .apply(lambda x: x.sort_index().shift())
                .rename(columns = {"fx_rtn": "lag_fx_rtn"})
                .reset_index()
                .drop(columns = ["group_var"])
                .dropna())
        
        df_combined = (df_fx
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"])
                .merge(right = df_spread, how = "inner", on = ["date", "variable", "etf_ticker"])
                .merge(right = df_lag_fx, how = "inner", on = ["date", "variable", "fx_ticker"])
                .merge(right = df_pcs,    how = "inner", on = ["date", "country"])
                .assign(group_var = lambda x: x.etf_ticker + "-" + x.fx_ticker + "-" + x.variable))
        
        exog_var = ["etf_rtn", "spread", "lag_fx_rtn", "PC1", "PC2", "PC3"]
        models   = self._ols_models(df_combined, exog_var)
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
def main() -> None: 
        
    factor_exposure = FactorExposure()
    factor_exposure.generate_etf_rtn_factor()
    factor_exposure.generate_etf_rtn_differential_factor()
    factor_exposure.generate_etf_rtn_and_differential_factor()
    factor_exposure.generate_all_factors()
    
if __name__ == "__main__": main()