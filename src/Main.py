# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:02:51 2026

@author: Diego
"""

from DataCollect             import DataCollect
from CollectIndividualStocks import IndividualStocks
from PrepData                import DataPrep
from FactorExposure          import FactorExposure
from Backtest                import Backtesting
from OptimizeSignal          import SignalOptimizer
from SignalGenerator         import SignalGenerator

def main() -> None: 

    # the module for collecting the data
    data_collect = DataCollect()
    data_collect.collect_etf()
    data_collect.collect_fx_carry_returns()
    data_collect.get_etf_factor_data()
    data_collect.get_index_factor_data()
    data_collect.get_curve_data()

    # the module for getting the individual stock data
    individual_stocks = IndividualStocks()
    individual_stocks.prep_holdings()
    individual_stocks.get_single_name_stocks()()

    # the module for preparing the data
    prep_data = DataPrep()
    prep_data.vol_fx_target_returns()
    prep_data.vol_etf_target_returns()
    prep_data.yield_curve_pca()

    # the module for getting the prepared data and testing factor exposure
    factor_exposure = FactorExposure()
    factor_exposure.generate_etf_rtn_factor()
    factor_exposure.generate_etf_rtn_differential_factor()
    factor_exposure.generate_etf_rtn_and_differential_factor()
    factor_exposure.generate_all_factors()

    # module for generating the signal 
    signal_generator = SignalGenerator()
    signal_generator.generate_spread_signal()
    signal_generator.generate_etf_signal()

    # module for optimizing the signals 
    optimizer = SignalOptimizer()
    optimizer.optimize_signal()

    # the module for backtesting the signals 
    backtesting = Backtesting()
    backtesting.get_backtest()

if __name__ == "__main__": main()