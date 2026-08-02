# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:58:37 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

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
        
    def _vol_target_rtn(
            self, 
            df_wider  : pd.DataFrame, 
            vol_target: float, 
            vol_window: int, 
            threshold : float) -> pd.DataFrame: 
        
        df_perf = (df_wider
                .apply(lambda x: x 
                       * (
                           self.vol_target / 
                           (x.ewm(span = self.vol_window, adjust = False).std() * np.sqrt(252))))
                .apply(lambda x: np.where(np.abs(x) > self.threshold, np.nan, x))
                .mean(axis = 1)
                .to_frame(name = "rtn")
                .assign(target = "perfect"))
        
        df_lagged = (df_wider
                .apply(lambda x: x 
                       * (
                           self.vol_target / 
                           (x.ewm(span = self.vol_window, adjust = False).std().shift() * np.sqrt(252))))
                .apply(lambda x: np.where(np.abs(x) > self.threshold, np.nan, x))
                .mean(axis = 1)
                .to_frame(name = "rtn")
                .assign(target = "lagged"))
        
        df_out = pd.concat([df_perf, df_lagged])
        return df_out
            
    def get_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Running backtest")
        
        opt_path  = os.path.join(self.data_path, "OptimizedSignal")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        opt_files = os.listdir(opt_path)
        
        for opt_file in opt_files:
            
            in_path  = os.path.join(opt_path, opt_file)
            out_path = os.path.join(self.back_path, opt_file) 
            
            if os.path.exists(out_path):
                if verbose: print("Already have {} backtest\n".format(opt_file.split(".")[0]))
                continue
            
            if verbose: print("Working on {}".format(opt_file.split(".")[0]))
            
            fx_ticker_map = (pd
                  .read_excel(io = tick_path, sheet_name = "TickerGuide")
                  .set_index("ticker")
                  .group
                  .to_dict())
            
            df_out = (pd
                    .read_parquet(path = in_path, engine = "pyarrow")
                    .rename(columns = {"group": "signal_group"})
                    .assign(
                        tmp_name = lambda x: x.signal_type + " " + x.sample_group,
                        group    = lambda x: x.fx_ticker.map(fx_ticker_map))
                    .groupby("tmp_name")
                    .apply(self._get_port_rtn, self.vol_target, self.vol_window, self.threshold)
                    .reset_index()
                    .assign(
                        signal_name  = opt_file.split(".")[0],
                        signal_type  = lambda x: x.tmp_name.str.split(" ").str[0],
                        sample_group = lambda x: x.tmp_name.str.split(" ").str[1])
                    .drop(columns = ["tmp_name"]))
            
            if verbose: print("Saving data\n")
            df_out.to_parquet(path = out_path, engine = "pyarrow")
            
def main() -> None: 
        
    backtesting = Backtesting()
    backtesting.get_backtest()
    
if __name__ == "__main__": main()