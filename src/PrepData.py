# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:43:20 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   sklearn.decomposition import PCA

from tqdm import tqdm

class DataPrep:
    
    def __init__(self) -> None: 
        
        self.cur_path    = os.getcwd()
        self.root_path   = os.path.abspath(os.path.join(self.cur_path, ".."))
        self.data_path   = os.path.join(self.root_path, "data")
        self.prep_path   = os.path.join(self.data_path, "PrepData")
        self.ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)
            
        self.vol_target  = 0.1
        self.vol_window  = 100
        self.outlier_rtn = 0.1
        
        self.window     = 10
        self.bad_fields = ["EBIT", "EBITDA", "GROSS_MARGIN", "IS_EPS"]

        
    def _get_vol_target(
            self, 
            df         : pd.DataFrame, 
            vol_target : float = 0.1, 
            vol_window : int = 100,
            outlier_rtn: float = 0.1) -> pd.DataFrame: 
        
        df_out = (df
                .sort_index()
                .assign(
                    px_rtn   = lambda x: x.PX_LAST.pct_change(),
                    vol      = lambda x: x.px_rtn.ewm(span = vol_window).std() * np.sqrt(252),
                    lag_rtn  = lambda x: x.px_rtn * (vol_target / x.vol.shift()),
                    perf_rtn = lambda x: x.px_rtn * (vol_target / x.vol))
                .assign(
                    lag_rtn  = lambda x: np.where(np.abs(x.lag_rtn) > outlier_rtn, np.nan, x.lag_rtn),
                    perf_rtn = lambda x: np.where(np.abs(x.perf_rtn) > outlier_rtn, np.nan, x.perf_rtn)))
        
        return df_out
            
    def get_vol_fx_target_returns(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Vol Targeted FX Returns")
        
        out_path = os.path.join(self.prep_path, "FXVolTargetedReturns.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        raw_files = ["FutPX", "FXCarryIndices"]
        df_lists  = []
        
        for file in raw_files: 
            
            in_path  = os.path.join(self.data_path, "RawData", file + ".parquet")
            df_input = (pd
                    .read_parquet(path = in_path, engine = "pyarrow")
                    .set_index("date")
                    .groupby("security")
                    .apply(self._get_vol_target, self.vol_target, self.vol_window, self.outlier_rtn)
                    .reset_index()
                    .drop(columns = ["PX_LAST", "vol"])
                    .rename(columns = {
                        "px_rtn"  : "none",
                        "lag_rtn" : "lagged",
                        "perf_rtn": "perfect"})
                    .melt(id_vars = ["security", "date"], var_name = "vol_target", value_name = "fx_rtn")
                    .dropna())
            
            df_lists.append(df_input)
            
        if verbose: print("Saving data\n")
        df_out = pd.concat(df_lists)
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_etf_returns(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Vol Targeted ETF Returns")
        
        out_path = os.path.join(self.prep_path, "ETFReturns.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path   = os.path.join(self.data_path, "RawData", "EquityPX.parquet")        
        df_out = (pd
                .read_parquet(path = path, engine = "pyarrow")
                .loc[lambda x: x.px_type == "Adj Close"]
                .drop(columns = ["px_type"])
                .set_index("date")
                .groupby("ticker")
                .apply(lambda x: x.sort_index().pct_change())
                .rename(columns = {"value": "rtn"})
                .reset_index())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _calc_rtn_spread(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame: 
        
        df_out = (df.reset_index().melt(
            id_vars    = ["date", ticker],
            var_name   = "for_ticker",
            value_name = "for_rtn").
            dropna().
            rename(columns = {ticker: "dom_rtn"}).
            assign(
                dom_ticker = ticker,
                rtn_spread = lambda x: x.dom_rtn - x.for_rtn))
        
        return df_out
        
    def _get_pcs(self, df: pd.DataFrame, pcs: int = 3) -> pd.DataFrame: 
        
        df_wider = (df
                .pivot(index = "date", columns = "security", values = "PX_LAST")
                .ffill()
                .dropna())
        
        pca_data = PCA(n_components = pcs).fit_transform(df_wider)
        df_pcs   = (pd
                .DataFrame(
                    data    = pca_data,
                    index   = df_wider.index,
                    columns = ["PC{}".format(i + 1) for i in range(pcs)])
                .reset_index()
                .melt(id_vars = "date"))
        
        return df_pcs
        
    def yield_curve_pca(
            self, 
            start_year: int = 2000, 
            pcs       : int = 3,
            verbose   : bool = True) -> None: 
        
        if verbose: print("Getting Yield Curve PCs")
        
        out_path = os.path.join(self.prep_path, "YieldCurvePCs.parquet")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        in_path      = os.path.join(self.data_path, "RawData", "RawYieldCurve.parquet")
        df_curve_raw = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .drop(columns = ["curve"])
                .assign(security = lambda x: x.security.str.split(" ").str[0]))

        
        df_date_selector = (df_curve_raw
                .drop(columns = ["PX_LAST"])
                .groupby(["security", "country"])
                ["date"]
                .agg("min")
                .to_frame(name = "date")
                .sort_values("date")
                .assign(year = lambda x: x.date.dt.year)
                .reset_index().
                loc[lambda x: x.year <= start_year]
                .drop(columns = ["date", "year"]))
        
        df_sliced = (df_date_selector
                .merge(right = df_curve_raw, how = "inner", on = ["security", "country"]))

        df_out = (df_sliced
                .groupby("country")
                .apply(self._get_pcs, pcs)
                .reset_index()
                .drop(columns = ["level_1"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _prep_etf_factor_data(self) -> None: 
        
        etf_path = os.path.join(self.data_path, "RawData", "ETFFactors.parquet")
        
        df_raw_etf = (pd
                .read_parquet(path = etf_path, engine = "pyarrow")
                .assign(security = lambda x: x.security.str.split(" ").str[0])
                .loc[lambda x: x.field != "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS"]
                .drop_duplicates())
        
        df_prep = (df_raw_etf
                .assign(group_var = lambda x: x.security + "-" + x.field)
                .pivot(index = "date", columns = "group_var", values = "value")
                .diff()
                .apply(lambda x: x / x.ewm(self.window, adjust = False).std())
                .cumsum()
                .reset_index()
                .melt(id_vars = "date", value_name = "value")
                .dropna()
                .assign(
                    calc       = "adjusted",
                    str_split  = lambda x: x.group_var.str.split("-"),
                    etf_ticker = lambda x: x.str_split.str[0],
                    field      = lambda x: x.str_split.str[1])
                .drop(columns  = ["group_var", "str_split"]))
        
        df_raw = (df_raw_etf
                .rename(columns = {"security": "etf_ticker"})
                .assign(calc = "raw"))
        
        df_out = (pd
                  .concat([df_prep, df_raw])
                  .assign(source = "etf_factor"))
        
        return df_out
    
    def _prep_index_factor_data(self) -> pd.DataFrame: 
        
        idx_path  = os.path.join(self.data_path, "RawData", "IndexFactors.parquet")
        tick_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        etf_renamer_ = (pd
                .read_excel(io = tick_path, sheet_name = "TickerGuide")
                .assign(
                    etf_benchmark = lambda x: x.etf_benchmark.str.split(" ").str[0],
                    etf_ticker    = lambda x: x.etf_ticker.str.split(" ").str[0])
                .drop(columns = ["CurveFile"])
                .dropna()
                .set_index("etf_benchmark")
                .etf_ticker
                .to_dict())
        
        spx_renamer_ = {"SPXT": "SPY"}
        etf_renamer  = {**etf_renamer_, **spx_renamer_}
        
        df_raw_idx = (pd
                .read_parquet(path = idx_path, engine = "pyarrow")
                .loc[lambda x: ~x.field.isin(self.bad_fields)]
                .assign(security = lambda x: x.security.str.split(" ").str[0])
                .set_index("date")
                .drop_duplicates()
                .assign(etf_ticker = lambda x: x.security.map(etf_renamer)))
        
        df_prep = (df_raw_idx
                .reset_index()
                .dropna()
                .assign(name = lambda x: x.etf_ticker + "-" + x.field)
                .pivot(index = "date", columns = "name", values = "value")
                .diff()
                .apply(lambda x: x / x.ewm(span = 10, adjust = False).std())
                .reset_index()
                .melt(id_vars = "date")
                .assign(
                    str_split  = lambda x: x.name.str.split("-"),
                    etf_ticker = lambda x: x.str_split.str[0],
                    field      = lambda x: x.str_split.str[1],
                    calc       = "adjusted")
                .drop(columns = ["str_split", "name"])
                .dropna())
        
        df_raw = (df_raw_idx
                  .assign(calc = "raw")
                  .reset_index()
                  .assign(etf_ticker = lambda x: x.security.map(etf_renamer)))
        
        df_out = (pd
                .concat([df_raw, df_prep])
                .assign(source = "index_factor")
                .drop(columns = ["security"]))
        
        return df_out
    
    def prep_factor_data(self, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Preparing Factor Data")
            
        out_path = os.path.join(self.data_path, "PrepData", "FactorData.parquet")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_etf = self._prep_etf_factor_data()
        df_idx = self._prep_index_factor_data()
        
        df_out = (pd
                  .concat([df_etf, df_idx])
                  .drop_duplicates()
                  .pivot(index = ["date", "calc", "field", "source"], columns = "etf_ticker", values = "value")
                  .reset_index()
                  .melt(id_vars = ["date", "calc", "field", "source", "SPY"])
                  .dropna()
                  .assign(spread = lambda x: x.value - x.SPY))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def prep_ml_factors(self, slice_year: int = 2018, verbose: bool = True) -> pd.DataFrame: 
        
        if verbose: print("Getting ML Factor Data")
        
        mom_path    = os.path.join(self.data_path, "Signals", "ReturnTrendSpreadSignal.parquet")
        factor_path = os.path.join(self.data_path, "Signals", "FactorSignal.parquet")
        ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        fx_path     = os.path.join(self.data_path, "PrepData", "FXVolTargetedReturns.parquet")
        out_path    = os.path.join(self.data_path, "PrepData", "MLFactors.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_ticker = (pd
                .read_excel(io = ticker_path, sheet_name = "TickerGuide")
                [["etf_ticker", "ticker", "Active"]]
                .loc[lambda x: x.Active == True]
                .drop(columns = ["Active"])
                .assign(etf_ticker = lambda x: x.etf_ticker.str.split(" ").str[0])
                .rename(columns = {"ticker": "fx_ticker"}))
        
        df_fx_rtn = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .loc[lambda x: x.vol_target == "perfect"]
                .drop(columns = ["vol_target"])
                .rename(columns = {"security": "fx_ticker"})
                .merge(right = df_ticker, how = "inner", on = ["fx_ticker"]))
        
        etf_tickers = df_ticker.etf_ticker.drop_duplicates().sort_values().to_list()

        df_momentum = (pd
            .read_parquet(path = mom_path, engine = "pyarrow")
            .drop(columns = ["signal", "window", "signal_name"])
            .rename(columns = {
                "ticker"    : "etf_ticker",
                "lag_signal": "momentum"})
            .dropna()
            .loc[lambda x: x.etf_ticker.isin(etf_tickers)])
        
        df_raw_signal = (pd
            .read_parquet(path = factor_path, engine = "pyarrow")
            .loc[lambda x: x.calc == "adjusted"]
            [["date", "field", "etf_ticker", "lag_signal"]]
            .dropna()
            .pivot(index = ["date", "etf_ticker"], columns = "field", values = "lag_signal")
            .reset_index()
            .merge(right = df_momentum, how = "inner", on = ["date", "etf_ticker"])
            .melt(id_vars = ["date", "etf_ticker"])
            .assign(
                year         = lambda x: x.date.dt.year,
                sample_group = lambda x: np.where(x.year <= slice_year, "in_sample", "out_sample"))
            .drop(columns = ["year"]))
        
        df_signal_selector = (df_raw_signal
                .loc[lambda x: x.sample_group == "in_sample"]
                .dropna()
                [["etf_ticker", "variable"]]
                .drop_duplicates())
        
        df_signal = (df_raw_signal
                .merge(right = df_signal_selector, how = "inner", on = ["etf_ticker", "variable"])
                .dropna()
                .merge(right = df_fx_rtn, how = "inner", on = ["date", "etf_ticker"]))

        if verbose: print("Saving data\n")
        df_signal.to_parquet(path = out_path, engine = "pyarrow")
    
    def _apply_fx(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        exch_path = os.path.join(self.data_path, "HoldingsGuide", "ExchangeMapper.csv")
        fx_path   = os.path.join(self.data_path, "RawData", "RawSpotFX.parquet")
        
        df_spot_fx = (pd
                .read_parquet(path = fx_path, engine = "pyarrow")
                .loc[lambda x: x.Price == "Close"]
                .drop(columns = ["Price"])
                .dropna()
                .assign(fx_ticker = lambda x: x.Ticker.str.split("=").str[0])
                .drop(columns = ["Ticker"])
                .rename(columns = {"value": "fx_val"}))
        
        df_exch = (pd
                .read_csv(filepath_or_buffer = exch_path)
                [["ShortName", "Currency"]]
                .rename(columns = {
                    "ShortName": "exch",
                    "Currency" : "fx_ticker"})
                .dropna())
        
        us_exchanges = (df_exch
                .loc[lambda x: x.fx_ticker == "USD"]
                .exch
                .to_list())
        
        df_us = (df
                .loc[lambda x: x.exch.isin(us_exchanges)]
                .assign(adj_val = lambda x: x.value)
                .reset_index())
        
        df_for = (df
                .reset_index()
                .loc[lambda x: ~x.exch.isin(us_exchanges)])
        

        df_for_adj = (df_for
                .merge(right = df_exch, how = "inner", on = ["exch"])
                .merge(right = df_spot_fx, how = "inner", on = ["date", "fx_ticker"])
                .assign(adj_val = lambda x: x.value * x.fx_val)
                .drop(columns = ["fx_val", "fx_ticker"]))
        
        df_out = (pd
                .concat([df_us, df_for_adj])
                .drop(columns = ["value"])
                .rename(columns = {"adj_val": "value"}))
        
        return df_out
        
        
    def prep_ml_single_name_momentum(self, verbose: bool = True) -> None:
        
        if verbose: print("Prepping ML Single Stock Momentum Data")
        
        spy_path    = os.path.join(self.data_path, "RawData", "EquityPX.parquet")
        single_path = os.path.join(self.data_path, "SingleNameStocks")
        files       = os.listdir(single_path)
        out_path    = os.path.join(self.data_path, "PrepData", "SingleNameData.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_spy   = (pd
         .read_parquet(path = spy_path, engine = "pyarrow")
         .loc[lambda x: x.ticker == "SPY"]
         .loc[lambda x: x.px_type == "Adj Close"]
         .set_index("date")
         [["value"]]
         .sort_index()
         .pct_change()
         .rename(columns = {"value": "spy_rtn"}))
        
        df_raw_px = (pd
                       .concat([
                           (pd
                            .read_parquet(
                                path   = os.path.join(single_path, file), 
                                engine = "pyarrow")
                            .assign(exch = file.split(".")[0])) 
                           for file in files])
                       .loc[lambda x: x.Price == "Adj Close"]
                       .dropna()
                       .drop(columns = ["Price"])
                       .rename(columns = {"Ticker": "stock_ticker"})
                       .assign(stock_ticker = lambda x: x.stock_ticker.astype(str))
                       .set_index("date"))
        
        df_adj = self._apply_fx(df_raw_px)
        
        df_raw_rtn = (df_adj
                       .set_index("date")
                       .groupby(["stock_ticker", "exch"])
                       .apply(lambda x: x.sort_index().value.pct_change())
                       .reset_index()
                       .rename(columns = {"value": "rtn"}))
        
        good_tickers = (df_raw_rtn
                .drop(columns = ["exch"])
                .dropna()
                [["date", "stock_ticker"]]
                .groupby("stock_ticker")
                ["date"]
                .agg(["min", "max", "count"])
                .sort_values("count")
                .rename(columns = {"min": "start_date"})
                .assign(year = lambda x: x.start_date.dt.year)
                .loc[lambda x: x.year < 2015]
                .index
                .to_list())
        
        df_px = (df_raw_rtn
                .loc[lambda x: x.stock_ticker.isin(good_tickers)])

        yf_guide_path = os.path.join(self.data_path, "HoldingsGuide", "YahooTickerMapper.csv")
        df_yf_guide   = (pd
                .read_csv(filepath_or_buffer = yf_guide_path)
                [["etf", "provider", "yf_ticker"]]
                .drop_duplicates()
                .rename(columns = {
                    "etf"      : "etf_ticker",
                    "yf_ticker": "stock_ticker"})
                .assign(stock_ticker = lambda x: x.stock_ticker.str.upper()))
        
        df_out = (df_px
                .merge(right = df_yf_guide, how = "left", on = ["stock_ticker"])
                .merge(right = df_spy     , how = "left", on = ["date"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
        
def main() -> None: 
        
    prep_data = DataPrep()
    #prep_data.get_vol_fx_target_returns()
    #prep_data.get_etf_returns()
    #prep_data.yield_curve_pca()
    #prep_data.prep_factor_data()
    #prep_data.prep_ml_factors()
    #prep_data.prep_ml_single_name_momentum()
    
if __name__ == "__main__": main()