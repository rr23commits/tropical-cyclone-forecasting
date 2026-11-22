### Held-out test (2020-01..2025-12, n=72; MAPE uses the 33 months with >=1 storm), sorted by RMSE

| Rank | Model | Family | RMSE | MAE | MAPE % (y>0) | sMAPE % | MASE | WAPE % | Bias |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Prophet | Additive/Bayesian | 1.431 | 1.030 | 51.4 | 133.8 | 0.945 | 61.8 | -0.093 |
| 2 | Prophet (tuned) | Additive/Bayesian | 1.459 | 1.043 | 52.2 | 133.8 | 0.956 | 62.6 | -0.080 |
| 3 | XGBoost+Prophet ensemble | Ensemble | 1.461 | 1.016 | 51.5 | 135.5 | 0.931 | 60.9 | -0.213 |
| 4 | LSTM (tuned) | Recurrent NN | 1.477 | 0.984 | 56.1 | 137.7 | 0.903 | 59.1 | -0.227 |
| 5 | GRU (tuned) | Recurrent NN | 1.502 | 0.998 | 57.1 | 138.2 | 0.915 | 59.9 | -0.291 |
| 6 | VARX (half-month) | Multivariate | 1.503 | 1.022 | 56.8 | 137.8 | 0.937 | 61.3 | -0.298 |
| 7 | XGBoost | Tree ensemble | 1.506 | 1.010 | 51.8 | 137.1 | 0.926 | 60.6 | -0.312 |
| 8 | XGBoost (tuned) | Tree ensemble | 1.523 | 0.976 | 53.5 | 137.8 | 0.895 | 58.5 | -0.320 |
| 9 | ARX-GARCH | Volatility | 1.536 | 1.012 | 56.7 | 141.1 | 0.928 | 60.7 | -0.443 |
| 10 | PatchTST | Transformer | 1.560 | 1.026 | 50.5 | 137.6 | 0.941 | 61.6 | -0.259 |
| 11 | LSTM | Recurrent NN | 1.947 | 1.254 | 60.1 | 138.5 | 1.150 | 75.2 | -0.271 |
| 12 | Seasonal naive | Baseline | 2.017 | 1.181 | 78.9 | 61.0 | 1.083 | 70.8 | +0.097 |
| 13 | SARIMA | Classical | 2.017 | 1.181 | 78.9 | 66.5 | 1.083 | 70.8 | +0.097 |
| 14 | AutoTS | AutoML | 2.017 | 1.097 | 48.9 | 57.4 | 1.006 | 65.8 | -0.375 |
| 15 | GRU | Recurrent NN | 2.050 | 1.401 | 59.6 | 141.3 | 1.285 | 84.1 | -0.301 |
| 16 | ARIMA | Classical | 2.392 | 1.528 | 94.5 | 104.0 | 1.401 | 91.7 | +0.000 |
| 17 | Holt-Winters | Classical | 2.606 | 2.093 | 92.4 | 134.3 | 1.920 | 125.6 | +1.690 |

### Validation (2016-01..2019-12), sorted by RMSE

| Rank | Model | Family | RMSE | MAE | MAPE % (y>0) | sMAPE % | MASE | WAPE % | Bias |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Prophet (tuned) | Additive/Bayesian | 0.905 | 0.603 | 39.5 | 122.1 | 0.553 | 41.4 | -0.130 |
| 2 | XGBoost+Prophet ensemble | Ensemble | 0.921 | 0.619 | 39.0 | 120.6 | 0.568 | 42.5 | -0.153 |
| 3 | XGBoost | Tree ensemble | 0.929 | 0.629 | 38.9 | 120.0 | 0.577 | 43.2 | -0.184 |
| 4 | Prophet | Additive/Bayesian | 0.935 | 0.631 | 39.5 | 121.5 | 0.578 | 43.2 | -0.115 |
| 5 | XGBoost (tuned) | Tree ensemble | 0.947 | 0.619 | 41.0 | 123.3 | 0.568 | 42.5 | -0.208 |
| 6 | ARX-GARCH | Volatility | 0.962 | 0.608 | 42.2 | 123.1 | 0.558 | 41.7 | -0.244 |
| 7 | VARX (half-month) | Multivariate | 0.968 | 0.622 | 40.5 | 121.4 | 0.570 | 42.6 | -0.173 |
| 8 | GRU (tuned) | Recurrent NN | 0.972 | 0.630 | 41.5 | 124.8 | 0.578 | 43.2 | -0.183 |
| 9 | LSTM (tuned) | Recurrent NN | 0.973 | 0.642 | 43.0 | 124.2 | 0.588 | 44.0 | -0.178 |
| 10 | SARIMA | Classical | 1.061 | 0.625 | 39.3 | 63.6 | 0.573 | 42.9 | -0.167 |
| 11 | Seasonal naive | Baseline | 1.061 | 0.625 | 39.3 | 59.4 | 0.573 | 42.9 | -0.167 |
| 12 | PatchTST | Transformer | 1.199 | 0.804 | 53.5 | 131.3 | 0.737 | 55.1 | -0.309 |
| 13 | AutoTS | AutoML | 1.346 | 0.896 | 54.0 | 78.7 | 0.821 | 61.4 | -0.062 |
| 14 | LSTM | Recurrent NN | 1.445 | 1.015 | 58.0 | 125.4 | 0.931 | 69.6 | -0.093 |
| 15 | GRU | Recurrent NN | 1.553 | 1.154 | 60.8 | 129.1 | 1.058 | 79.1 | -0.146 |
| 16 | ARIMA | Classical | 1.837 | 1.208 | 98.9 | 97.4 | 1.108 | 82.9 | -0.000 |
| 17 | Holt-Winters | Classical | 2.223 | 1.749 | 112.5 | 125.3 | 1.604 | 120.0 | +1.530 |
