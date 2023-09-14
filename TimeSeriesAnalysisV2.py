
# Updated Stock Market Prediction using ARIMA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def test_stationarity(timeseries):
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1] <= 0.05

def apply_ARIMA(train, test):
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, forecast)
    return mse

data = pd.read_csv('AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
ts_data = data['Close']

is_stationary = test_stationarity(ts_data)

if not is_stationary:
    ts_data_diff = ts_data.diff().dropna()
    is_stationary = test_stationarity(ts_data_diff)

train_size = int(len(ts_data_diff) * 0.8)
train, test = ts_data_diff[0:train_size], ts_data_diff[train_size:]
mse = apply_ARIMA(train, test)
print('MSE:', mse)
