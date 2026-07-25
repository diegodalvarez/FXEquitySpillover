# FX Equity Spillover

This repo explores trading strategies with FX Futures based on the equity differentials between US and each FX future's respective equity. The main idea is that excess foreign equity returns leads to increased FX value. 

Within the two notebooks ```1SignalStrategies.ipynb``` and ```2CrossSectionalStrategies.ipynb``` The following strategies are researched

| Strategy        | Vol. Target   |   Adj Sharpe |   Raw Sharpe |
|:----------------|:--------------|-------------:|-------------:|
| Combo           | Lagged        |        1.109 |        1.109 |
| Combo           | Perfect       |        1.175 |        1.175 |
| Resid Port      | Lagged        |        0.514 |        0.474 |
| Resid Port      | Perfect       |        0.571 |        0.527 |
| Z-Score Port    | Lagged        |        1.141 |        1.141 |
| Z-Score Port    | Perfect       |        1.193 |        1.193 |
| Cross Sectional | Lagged        |        0.745 |              | 
| Cross Sectional | Perfect       |        0.688 |              |

## Writeup
|         | PDF          |
|---------|---------------------|
| Technical writeup containing methodology & results | <a href="https://github.com/diegodalvarez/FXEquitySpillover/raw/main/FXEquitySpilloverWriteup.pdf">![PDF](https://img.icons8.com/ios-filled/50/000000/pdf.png)</a> |