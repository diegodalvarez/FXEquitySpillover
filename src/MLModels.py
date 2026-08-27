# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 02:48:23 2026

@author: Diego
"""

import os
import pickle
import pandas as pd

from xgboost import XGBRegressor

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression

class MLModels:
    
    def __init__(self) -> None:
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.ml_path   = os.path.join(self.data_path, "MLModels")
        
        if not os.path.exists(self.ml_path): 
            os.makedirs(self.ml_path)
            
        self.models = {
            "linear": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator = True)),
                ("scaler" , StandardScaler()),
                ("model"  , LinearRegression())]),
        
            "random_forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator = True)),
                ("scaler" , StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=100,
                    max_features=0.5,
                    max_samples=0.8,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            
            "elastic_net": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator = True)),
                ("scaler" , StandardScaler()),
                ("model", ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=10000
                ))
            ]),
                        
            "ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator = True)),
                ("scaler" , StandardScaler()),
                ("model", Ridge(alpha=10.0))
            ]),
            
            "polynomial": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator = True)),
                ("scaler" , StandardScaler()),
                ("poly", PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )),
                ("model", Ridge(alpha=100))
            ]),
                    
            "xgboost": Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )),
                
                ("model", XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=2,
                    min_child_weight=30,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=5.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        }
        
    def fit_models(self, verbose: bool = True) -> None:
        
        signal_path = os.path.join(self.data_path, "PrepData", "MLFactors.parquet")
        out_path    = os.path.join(self.data_path, "MLModels", "Models.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_signal = (pd
                .read_parquet(path = signal_path, engine = "pyarrow"))
        
        fitted_models = {}
        
        tickers = df_signal.fx_ticker.drop_duplicates().sort_values().to_list()
        for ticker in tickers: 
            
            df_tmp_signal = (df_signal
                    .loc[lambda x: x.fx_ticker == ticker]
                    .drop(columns = ["fx_ticker"])
                    .dropna()
                    .pivot(
                        index   = ["date", "etf_ticker", "sample_group", "fx_rtn"], 
                        columns = "variable", 
                        values  = "value")
                    .reset_index()
                    .set_index("date"))
            
            df_input = (df_tmp_signal
                    .loc[lambda x: x.sample_group == "in_sample"]
                    .drop(columns = ["sample_group", "etf_ticker"]))
            
            X = df_input.drop(columns = ["fx_rtn"])
            y = df_input["fx_rtn"]
            
            fitted_models[ticker] = {}
            
            for model_name, model in self.models.items():
                
                print(f"{ticker} - {model_name}")
                fitted_model = clone(model)
                fitted_model.fit(X,y)
                fitted_models[ticker][model_name] = fitted_model
                
        if verbose: 
            print("Saving data\n")
                
        with open(out_path, "wb") as f:
            pickle.dump(fitted_models, f)
            
    def _get_metrics(self, df: pd.DataFrame, X_cols: list, model) -> dict: 
        
        X, y  = df[X_cols], df["fx_rtn"]
        y_hat = model.predict(X)
        rmse  = root_mean_squared_error(y, y_hat)
        r2    = r2_score(y, y_hat)
        
        out_dict = {
            "rmse": rmse,
            "r2"  : r2}
        
        return out_dict
            
    def eval_models(self, verbose: bool = True) -> None:
        
        if verbose: 
            print("Getting Model Metrics")
        
        model_path = os.path.join(self.data_path, "MLModels", "Models.pkl")
        sig_path   = os.path.join(self.data_path, "PrepData", "MLFactors.parquet")
        out_path   = os.path.join(self.data_path, "MLModels", "ModelMetrics.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        with open(model_path, "rb") as f: models = pickle.load(f)
        
        df_signal = pd.read_parquet(path = sig_path, engine = "pyarrow")
        
        results = []
        
        for model_name, tmp_models in models.items():
    
            df_tmp_signal = (
                df_signal
                .loc[lambda x: x.fx_ticker == model_name]
                .drop(columns=["etf_ticker", "fx_ticker"]))
    
            X_cols = [
                col
                for col in df_tmp_signal.columns
                if col not in ["fx_rtn", "sample_group"]]
    
            for model_type, model in tmp_models.items():
    
                for sample_group in [
                    "in_sample",
                    "out_sample",
                    "full_sample"]:
    
                    if sample_group == "full_sample":
    
                        df_tmp = (
                            df_tmp_signal
                            .drop(columns=["sample_group"])
                            .pivot(index = ["date", "fx_rtn"], columns = "variable", values = "value")
                            .reset_index()
                            .set_index("date"))
                        
                        X_cols = [
                            col for col in df_tmp.columns.to_list()
                            if col != "fx_rtn"]
    
                    else:
    
                        df_tmp = (
                            df_tmp_signal
                            .loc[lambda x: x.sample_group == sample_group]
                            .drop(columns=["sample_group"])
                            .pivot(index = ["date", "fx_rtn"], columns = "variable", values = "value")
                            .reset_index()
                            .set_index("date"))
    
                        X_cols = [
                            col for col in df_tmp.columns.to_list()
                            if col != "fx_rtn"]
    
                    metrics = self._get_metrics(
                        df     = df_tmp,
                        X_cols = X_cols,
                        model  = model)
    
                    results.append({
                        "fx_ticker"    : model_name,
                        "model"        : model_type,
                        "sample_group" : sample_group,
                        **metrics})
                    
        df_out = pd.DataFrame(results)
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
ml_models = MLModels()
ml_models.fit_models()
ml_models.eval_models()