# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:27:27 2026

@author: Diego
"""

import os
import pandas as pd

class FactorExposureTools:
    
    def __init__(self) -> None: 
        
        self.src_path    = os.getcwd()
        self.repo_path   = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
            
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
            
            sec, target, rtn_type, etf = key.split("-")
            return_type_mapper         = (df_ticker
                .set_index("fx_ticker")
                .return_type
                .to_dict())
        
            scaler_name = return_type_mapper[sec]
            
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