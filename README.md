# FX Equity Spillover

This repo explores trading strategies with FX futures based on the equity differentials between the US and each FX future's respective equity market. The main idea is that excess foreign equity returns lead to increased FX value.

## Notebooks

0. [`0DataComparison.ipynb`](0DataComparison.ipynb) — Examine Data.
1. [`1FactorModel.ipynb`](1FactorModel.ipynb) — Model and examine relationship between FX returns and relevant signals and factors
2. [`2ReturnsMomentumModel.ipynb`](2ReturnsMomentumModel.ipynb) — Initial model that trades FX returns based on sovereign equity returns differential momentum
3. [`3FactorZoo.ipynb`](3FactorZoo.ipynb) — Expanding to other factors beyond momentum
4. [`4MachineLearningApproach.ipynb`](4MachineLearningApproach.ipynb) — Using Machine Learning to find non-linear alpha after testing OLS models
5. [`5SingleNameStock.ipynb`](5SingleNameStock.ipynb) — Using Single Name stocks as opposed to ETFs.

## Writeup

| | PDF |
|----------------|---------------------|
| Technical Writeup containing methodology & results | <a href="https://github.com/diegodalvarez/FXEquitySpillover/blob/main/FXEquitySpilloverWriteup.pdf">![image](https://img.icons8.com/ios-filled/50/000000/pdf.png)</a> |
