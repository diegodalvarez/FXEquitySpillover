# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:56:23 2026

@author: Diego
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

class FactorSignal:
    
    def __init__(self) -> None: 
        
        self.cur_path    = os.getcwd()
        self.root_path   = os.path.abspath(os.path.join(self.cur_path, ".."))
        self.data_path   = os.path.join(self.root_path, "data")
        self.ticker_path = os.path.join(self.data_path, "FXTickerGuide.xlsx")
        
        self.df_ticker   = pd.read_excel(io = self.ticker_path, sheet_name = "TickerGuide")
        self.df_variable = pd.read_excel(io = self.ticker_path, sheet_name = "VariableMapper")
        
    def