# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 00:32:08 2026

@author: Diego
"""

import os
import pandas as pd
import datetime as dt
import yfinance as yf

class IndividualStocks:
    
    def __init__(self) -> None:
    
        self.src_path    = os.getcwd()
        self.repo_path   = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
        self.single_path = os.path.join(self.data_path, "SingleNameStocks")
        
        if not os.path.exists(self.single_path):
            os.makedirs(self.single_path)
        
        self.vaneck_col_mapper = {
            'Ticker'          : 'ticker',
            'Name'            : 'name',
            'Sector'          : 'sector',
            'Asset Class'     : 'asset_class',
            'Market Value'    : 'market_value',
            'Weight (%)'      : 'weight',
            'Notional Value'  : 'notional_weight',
            'Quantity'        : 'quantity',
            'Price'           : 'price',
            'Location'        : 'location',
            'Exchange'        : 'exchange',
            'Currency'        : 'currency',
            'FX Rate'         : 'fx_rate',
            'Market Currency' : 'market_currency'}
        
        self.msci_col_mapper = {
            '% of Net Assets'  : 'pct_net_asset',
            'Ticker'           : 'ticker',
            'Name'             : 'name',
            'SEDOL'            : 'sedol',
            'Market Price ($)' : 'mkt_price',
            'Shares Held'      : 'shares_held',
            'Market Value ($)' : 'mkt_val'}
        
        self.start_date = dt.date(year = 1990, month = 1, day = 1)
        self.end_date   = dt.date(year = 2026, month = 8, day = 1)
        
    def prep_holdings(self, verbose: bool = True) -> None: 
        
        if verbose: print("Preparing holdings")
        
        path     = os.path.join(self.data_path, "HoldingsGuide")
        out_path = os.path.join(path, "HoldingsCombined.csv")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
    
        # for VanEck
        vaneck_files = ([
            path for path in os.listdir(path)
            if path.split(".")[0].split("_")[-1] == "holdings"])
        
        msci_files = ([
            path for path in os.listdir(path)
            if path.split("-")[-1].split("_")[0] == "holdings"])
    
        df_lists = []
    
        for vaneck_file in vaneck_files: 
            
            in_path = os.path.join(path, vaneck_file)
            cols    = list(self.vaneck_col_mapper.keys())
            ticker  = vaneck_file.split("_")[0]
            
            df_add = (pd
                    .read_csv(
                        filepath_or_buffer = in_path, 
                        skiprows           = 8)
                    [cols]
                    .rename(columns = self.vaneck_col_mapper)
                    .loc[lambda x: x.asset_class == "Equity"]
                    .melt(id_vars = ["ticker"])
                    .assign(etf = ticker))
            
            df_lists.append(df_add)
        
        df_vaneck = (pd
                     .concat(df_lists)
                     .assign(provider = "VanEck"))
        
        df_lists = []
        for msci_file in msci_files:
            
            in_path = os.path.join(path, msci_file)
            ticker  = msci_file.split("_")[0]
            df_add  = (pd
                    .read_csv(filepath_or_buffer = in_path, skiprows = 2)
                    .rename(columns = self.msci_col_mapper)
                    .assign(a = lambda x: x.pct_net_asset.str[0:3].str.lower())
                    .loc[lambda x: x.a != "the"]
                    .drop(columns = ["a"])
                    .melt(id_vars = ["ticker"])
                    .assign(etf = ticker))
            
            df_lists.append(df_add)
            
        df_msci = (pd
                   .concat(df_lists)
                   .assign(provider = "MSCI"))
        
        df_out = (pd.concat([df_vaneck, df_msci]))
        if verbose: print("Saving data\n")
        df_out.to_csv(out_path)
        
    def get_single_name_stocks(self, verbose: bool = True) -> None: 
        
        holding_path = os.path.join(self.data_path, "HoldingsGuide", "YahooTickerMapper.csv")
        df_tickers   = (pd
                .read_csv(filepath_or_buffer = holding_path)
                [["value", "yf_ticker"]]
                .drop_duplicates()
                .dropna())
        
        exchange_path  = os.path.join(self.data_path, "HoldingsGuide", "ExchangeMapper.csv")
        exchange_namer = (pd
                .read_csv(filepath_or_buffer = exchange_path)
                .set_index("RawName")
                .ShortName
                .to_dict())
        
        exchanges = df_tickers.value.drop_duplicates().sort_values().to_list()
        for exchange in exchanges: 
            
            if verbose: print("Working on {}".format(exchange))

            file_name = exchange_namer[exchange]
            tmp_path  = os.path.join(self.single_path, file_name + ".parquet")
            
            if os.path.exists(tmp_path):
                if verbose: print("Already have data\n")
                continue

            tickers = (df_tickers
                    .loc[lambda x: x.value == exchange]
                    .yf_ticker
                    .to_list())
            
            df_out = (yf
                    .download(
                        tickers     = tickers, 
                        start       = self.start_date, 
                        end         = self.end_date, 
                        auto_adjust = False)
                    .reset_index()
                    .melt(id_vars = [("Date", "")])
                    .rename(columns = {("Date", ""): "date"}))
            
            if verbose: print("Saving data\n")
            df_out.to_parquet(path = tmp_path, engine = "pyarrow")
            
    def get_fx(self, verbose: bool = True) -> None:
        
        if verbose: 
            print("Getting Raw FX")
        
        exch_path = os.path.join(self.data_path, "HoldingsGuide", "ExchangeMapper.csv")
        out_path  = os.path.join(self.data_path, "RawData", "RawSpotFX.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        raw_fxs = (pd
                .read_csv(filepath_or_buffer = exch_path)
                .Currency
                .drop_duplicates()
                .dropna()
                .to_list())
        
        fx_tickers = ["{}=X".format(ticker) for ticker in raw_fxs]
        df_out = (yf
                .download(
                    tickers     = fx_tickers, 
                    start       = self.start_date, 
                    end         = self.end_date, 
                    auto_adjust = False)
                .reset_index()
                .melt(id_vars = [("Date", "")])
                .rename(columns = {("Date", ""): "date"}))
    
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
    
def main() -> None: 
    
    individual_stocks = IndividualStocks()
    #individual_stocks.prep_holdings()
    #individual_stocks.get_single_name_stocks()
    individual_stocks.get_fx()
    
if __name__ == "__main__": main()