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
            
    def _ols_models(self, df: pd.DataFrame) -> dict:
        
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
                    exog  = sm.add_constant(df_tmp.etf_rtn))
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
        
        models = self._ols_models(df_combined)
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
def main() -> None: 
        
    factor_exposure = FactorExposure()
    factor_exposure.generate_etf_rtn_factor()
    
if __name__ == "__main__": main()