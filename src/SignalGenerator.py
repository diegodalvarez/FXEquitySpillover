# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:44:26 2026

@author: Diego
"""

import os
import pandas as pd

class SignalGenerator:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.sig_path  = os.path.join(self.data_path, "Signals")
        
        if not os.path.exists(self.sig_path):
            os.makedirs(self.sig_path)
            
    def _lag_signal(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df
                  .sort_index()
                  .assign(lag_signal = lambda x: x.signal.shift()))
        
        return df_out
            
    def generate_spread_signal(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Returns Differential Spread")
        
        in_path  = os.path.join(self.data_path , "PrepData", "ETFVolTargetedReturns.parquet")
        out_path = os.path.join(self.sig_path, "ReturnSpreadSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_zscore = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .drop(columns = ["PX_LAST", "vol"])
                .melt(id_vars = ["date", "ticker"])
                .pivot(index = ["date", "variable"], columns = "ticker", values = "value")
                .reset_index()
                .melt(id_vars = ["date", "variable", "SPY"])
                .dropna()
                .assign(spread = lambda x: x.value - x.SPY)
                .pivot(index = "date", columns = ["variable", "ticker"], values = "spread")
                .apply(lambda x: 
                       (x - x.ewm(span = window, adjust = False).mean()) / 
                       x.ewm(span = window, adjust = False).std())
                .reset_index()
                .melt(id_vars = [("date", "")])
                .rename(columns = {
                    ("date", ""): "date",
                    "value"     : "signal"})
                .assign(group_var = lambda x: x.variable + " " + x.ticker)
                .set_index("date")
                .groupby("group_var")
                .apply(self._lag_signal)
                .reset_index()
                .drop(columns = ["group_var"])
                .dropna()
                .assign(window = window))
        
        if verbose: print("Saving data\n")
        df_zscore.to_parquet(path = out_path, engine = "pyarrow")
        
    def generate_etf_signal(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Returns Z-Score")
        
        in_path  = os.path.join(self.data_path , "PrepData", "ETFVolTargetedReturns.parquet")
        out_path = os.path.join(self.sig_path, "ReturnsSignal.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        in_path = os.path.join(self.data_path, "PrepData", "ETFVolTargetedReturns.parquet")
        df_zscore = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .drop(columns = ["PX_LAST", "vol", "lag_rtn"])
                .melt(id_vars = ["date", "ticker"])
                .dropna()
                .pivot(index = "date", columns = ["variable", "ticker"], values = "value")
                .apply(lambda x: 
                       (x - x.ewm(span = window, adjust = False).mean()) / 
                       x.ewm(span = window, adjust = False).std())
                .reset_index()
                .melt(id_vars = [("date", "")])
                .rename(columns = {
                    ("date", ""): "date",
                    "value"     : "signal"})
                .dropna()
                .assign(group_var = lambda x: x.variable + " " + x.ticker)
                .groupby("group_var")
                .apply(self._lag_signal)
                .reset_index()
                .drop(columns = ["group_var"])
                .assign(window = window))
            
        if verbose: print("Saving data\n")
        df_zscore.to_parquet(path = out_path, engine = "pyarrow")
            
signal_generator = SignalGenerator()
#signal_generator.generate_spread_signal()
signal_generator.generate_etf_signal()