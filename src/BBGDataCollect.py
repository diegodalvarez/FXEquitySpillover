# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:21:23 2026

@author: Diego
"""

import os
import pandas as pd
import yfinance as yf
import datetime as dt

class BBGDataCollect:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")        
        self.raw_data  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: 
            os.makedirs(self.data_path)
            
        if os.path.exists(self.raw_data) == False: 
            os.makedirs(self.raw_data)
            
        self.start_date = dt.date(year = 1900, month = 1, day = 1)
        self.end_date   = dt.date(year = 2026, month = 7, day = 20) 
        
        self.bdp_path  = r"A:\2026BlpAdHocData"
        self.etf_path1 = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260422AprilDataCollect (passed)"
        
    def collect_etf(self, verbose: bool = True) -> None: 
        
        if verbose: print("Collecting ETF Data")
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.raw_data, "EquityPX.parquet")
        
        etf_tickers_ = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .assign(front = lambda x: x.etf_ticker.str.split(" ").str[0])
                .front
                .dropna()
                .sort_values()
                .drop_duplicates()
                .to_list())
        
        etf_tickers = etf_tickers_ + ["SPY"]
        
        if os.path.exists(out_path) == True: 
            if verbose: print("Already have data\n")
            return None
        
        df_out = (yf
                .download(
                    tickers     = etf_tickers,
                    start       = self.start_date,
                    end         = self.end_date,
                    auto_adjust = False)
                .reset_index()
                .melt(id_vars = [("Date", "")])
                .rename(columns = {
                    ("Date", ""): "date",
                    "Price"     : "px_type",
                    "Ticker"    : "ticker"})
                .dropna())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
    
    def _get_carry_rtn(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (df
                .sort_values("date")
                .assign(rtn = lambda x: x.PX_LAST.pct_change())
                .dropna())
        
        return df_out
        
    def collect_fx_carry_returns(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Carry Indices")
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        master_path = os.path.join(self.bdp_path, "BloombergMasterTickers.xlsx")
        out_path    = os.path.join(self.raw_data, "FXCarryIndices.parquet")
        
        if os.path.exists(out_path): 
            if verbose: print("Already have Carry Indices\n")
            return None
        
        tickers = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                .loc[lambda x: x.group.isin(["em_carry", "g10_carry"])]
                .ticker
                .sort_values()
                .drop_duplicates()
                .to_list())
        
        files = (pd
                .read_excel(io = master_path, sheet_name = "PX")
                .loc[lambda x: x.ticker.isin(tickers)]
                .file
                .drop_duplicates()
                .to_list())
        
        paths = [
            os.path.join(self.bdp_path, "Combined", "PX", file + ".parquet")
            for file in files]
        
        df_lists = []
        
        for path in paths: 
            
            df_tmp = (pd
                    .read_parquet(path = path, engine = "pyarrow")
                    .assign(file = path.split("\\")[-1].split(".")[0]))
            
            df_lists.append(df_tmp)
            
        df_combined      = pd.concat(df_lists)
        df_file_selector = (df_combined
                .drop(columns = ["PX_LAST"])
                .groupby(["security", "file"])
                ["date"]
                .agg(["min", "max"])
                .drop(columns = ["max"]) # all ended up having similar end dates
                .reset_index()
                .rename(columns = {"min": "min_date"})
                .groupby("security")
                .apply(lambda x: x.loc[lambda x: x.min_date == x.min_date.min()])
                .reset_index()
                [["security", "file"]])
                
        df_out = (df_combined
                .merge(right = df_file_selector, how = "inner", on = ["security", "file"])
                .drop(columns = ["file"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_etf1(self, etf_tickers: list) -> pd.DataFrame: 
         

        eq_path  = os.path.join(self.bdp_path, "Combined", "Equity")
        etf_path = os.path.join(eq_path, "EquityETFs.parquet")
        
        df_etf1 = (pd
                .read_parquet(path = etf_path, engine = "pyarrow")
                .loc[lambda x: x.security.isin(etf_tickers)])
        
        return df_etf1
    
    def _get_etf2(self, tickers: list) -> pd.DataFrame: 
        
        files = (pd
                .DataFrame(data = {"path": os.listdir(self.etf_path1)})
                .assign(ending = lambda x: x.path.str.split(".").str[-1])
                .loc[lambda x: x.ending == "parquet"]
                .assign(file = lambda x: x.path.str.split(".").str[0].str[-4:])
                .loc[lambda x: x.file.isin(["FLDs", "FLDS"])]
                .path
                .to_list())
        
        paths  = [os.path.join(self.etf_path1, file) for file in files]
        df_out = (pd
                .read_parquet(path = paths, engine = "pyarrow")
                .melt(id_vars = ["date", "security"], var_name = "field")
                .dropna()
                .loc[lambda x: x.security.isin(tickers)])
        
        return df_out
        
    def get_etf_factor_data(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Equity ETF Factor Data")
        
        out_path = os.path.join(self.raw_data, "ETFFactors.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
    
        ticker_path  = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker    = pd.read_excel(io = ticker_path, sheet_name = "TickerGuide")
        etf_tickers_ = df_ticker.etf_ticker.drop_duplicates().to_list()
        etf_tickers  = etf_tickers_ + ["SPY US Equity"]

        df1         = self._get_etf1(etf_tickers)
        df2         = self._get_etf2(etf_tickers)
        df_combined = pd.concat([df1, df2])

        df_combined.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_index_factor_data(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Index Factor Data")
        
        eq_path     = os.path.join(self.bdp_path, "Combined", "Equity")
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        out_path    = os.path.join(self.raw_data, "IndexFactors.parquet")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None

        index_tickers_ = (pd
                      .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                      .etf_benchmark
                      .drop_duplicates()
                      .to_list())
        
        index_tickers = index_tickers_ + ["SPXT Index"]
        
        index_path1 = os.path.join(eq_path, "EquityIndices1.parquet")
        index_path2 = os.path.join(eq_path, "EquityIndices2.parquet")
        
        df1 = pd.read_parquet(path = index_path1, engine = "pyarrow")
        df2 = (pd
               .read_parquet(path = index_path2, engine = "pyarrow")
               .rename(columns = {"variable": "field"}))
        
        files = (pd
              .DataFrame({"file": os.listdir(self.etf_path1)})
              .assign(ending = lambda x: x.file.str.split(".").str[1])
              .loc[lambda x: x.ending == "parquet"]
              .assign(tmp = lambda x: x.file.str.split(".").str[0].str[-4:])
              .loc[lambda x: x.tmp.isin(["Real", "FLDS", "FLDs"])]
              .file
              .drop_duplicates()
              .to_list())
        
        paths = [os.path.join(self.etf_path1, file) for file in files]
        df3   = (pd
                .read_parquet(path = paths, engine = "pyarrow")
                .melt(id_vars = ["date", "security"], var_name = "field")
                .dropna())
        
        df_combined = (pd.concat(
            [df1, df2, df3])
            .loc[lambda x: x.security.isin(index_tickers)])
        
        if verbose: print("Saving data\n")
        df_combined.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_curve_data(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Curve Data")
        
        out_path = os.path.join(self.raw_data, "RawYieldCurve.parquet")
        if os.path.exists(out_path):
           if verbose: print("Already have data\n")
           return None
        
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker   = pd.read_excel(io = ticker_path, sheet_name = "TickerGuide")
        
        country_mapper = (df_ticker
                [["CurveFile", "country"]]
                .dropna()
                .set_index("CurveFile")
                .country
                .to_dict())
        
        files   = df_ticker.CurveFile.dropna().drop_duplicates().to_list()
        df_list = [] 
        df_out  = pd.DataFrame()
        
        for file in files: 
            
            tmp_path = os.path.join(self.bdp_path, "Combined", "PX", file + ".parquet")
            df_tmp   = (pd
                    .read_parquet(path = tmp_path, engine = "pyarrow")
                    .assign(
                        curve   = file,
                        country = lambda x: x.curve.map(country_mapper)))
            
            df_list.append(df_tmp)
            
        if verbose: print("Saving data\n")
            
        df_out = pd.concat(df_list)
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 

    data = BBGDataCollect()
    data.collect_etf()
    data.collect_fx_carry_returns()
    data.get_etf_factor_data()
    data.get_index_factor_data()
    data.get_curve_data()
    
if __name__ == "__main__": main()