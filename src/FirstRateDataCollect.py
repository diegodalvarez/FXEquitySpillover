# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 00:19:19 2026

@author: Diego
"""

import os
import zipfile
import pandas as pd

class FirstRateData:
    
    def __init__(self) -> None: 
        
        self.src_path   = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.fdata_path = os.path.join(self.data_path, "FirstRateData")
        
        if not os.path.exists(self.fdata_path):
            os.makedirs(self.fdata_path)
        
        self.etf_path = r"G:\FirstRateData\ETFData"
        
        self.etf_columns = ["datetime", "Open", "High", "Low", "Close", "Volume"]
        
    def get_etf_rtn(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting First Rate Hourly ETF Prices")
        
        out_path = os.path.join(self.fdata_path, "RawData.parquet")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        guide_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        df_ticker  = (pd
                .read_excel(
                    io         = guide_path, 
                    sheet_name = "TickerGuide")
                [["etf_ticker"]]
                .drop_duplicates()
                .dropna()
                .assign(
                    ticker = lambda x: x.etf_ticker.str.split(" ").str[0],
                    letter = lambda x: x.ticker.str[0]))
        
        letters   = df_ticker.letter.drop_duplicates().sort_values().to_list()
        etf_files = []
        etf_paths = []
        
        for letter in letters: 
            
            path = os.path.join(self.etf_path, "etf_{}_full_1hour_adj_splitdiv.zip".format(letter))
            with zipfile.ZipFile(path, "r") as z: files = z.namelist()
            
            tmp_tickers = (df_ticker
                    .loc[lambda x: x.letter == letter]
                    .ticker
                    .drop_duplicates()
                    .sort_values()
                    .to_list())
            
            add_etf_files = (pd
                    .DataFrame(data = {"file": files})
                    .assign(ticker = lambda x: x.file.str.split("_").str[0])
                    .loc[lambda x: x.ticker.isin(tmp_tickers)]
                    .file
                    .drop_duplicates()
                    .sort_values()
                    .to_list())
            
            add_etf_paths = [path for tmp_ in add_etf_files]
            
            etf_files += add_etf_files
            etf_paths += add_etf_paths
            
        df_lists = []
            
        for path, file in zip(etf_paths, etf_files):
            with zipfile.ZipFile(path, "r") as z:
                with z.open(file) as f:
        
                    df_tmp = (pd
                              .read_csv(
                                  filepath_or_buffer = f,
                                  header             = None,
                                  names              = self.etf_columns)
                              .melt(id_vars = "datetime")
                              .assign(ticker = file.split("_")[0]))
                    
                    df_lists.append(df_tmp)
                    
        if verbose: print("Saving data\n")
        
        df_out = pd.concat(df_lists)
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
            
    frate_data = FirstRateData()
    frate_data.get_etf_rtn()
    
if __name__ == "__main__": main()