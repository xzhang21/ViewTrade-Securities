# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:02:12 2019
https://medium.com/@hsahu/stock-prediction-with-xgboost-a-technical-indicators-approach-5f7e5940e9e3
@author: Lauren
"""



from pandas import Series
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from KafkaProcess import Consumer,KafkaProcess

# --------- import data from csv  ---------
series = Series.from_csv('XOM.csv', header=0)# if header=0, skip 0 row

N_test = 100
series = np.log(series) - np.log(series.shift(1)) # log return of stock price
original = series
test = series.tail(N_test)
series = series[:-N_test]
series = series.dropna()

# Moving Averages (SMA & EWMA)

# Simple Moving Average 
def SMA(data, days): 
    sma = pd.Series(pd.rolling_mean(data['Close'], days), name = 'SMA_' + str(days))
    data = data.join(sma) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, days):
    ema = pd.Series(pd.ewma(data['Close'], span = days, min_periods = days - 1), 
    name = 'EWMA_' + str(days))
    data = data.join(ema) 
    return data
days_list = [10,50,100,200]
for days in days_list:
    data = SMA(data,days)
    data = EWMA(data,days)
    
#Bollinger Bands (UpperBB & LowerBB)
def bbands(data, window=days):
    MA = data.Close.rolling(window=days).mean()
    SD = data.Close.rolling(window=days).std()
    data['UpperBB'] = MA + (2 * SD) 
    data['LowerBB'] = MA - (2 * SD)
    return data
days = 50
data = bbands(data, days)


# Force Index (ForceIndex)
def ForceIndex(data, days): 
    FI = pd.Series(data['Close'].diff(days) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data
days = 1
data = ForceIndex(data,days)

# Commodity Channel Index (CCI)
def CCI(data, days): 
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - pd.rolling_mean(TP, days)) / (0.015 * pd.rolling_std(TP, days)), 
    name = 'CCI')
    data = data.join(CCI)
    return data
days = 20
data = CCI(data, days)

# Ease Of Movement (EVM)
def EVM(data, days): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(pd.rolling_mean(EVM, days), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data 
days = 14
data = EVM(data, days)

# Rate of Change (ROC)
def ROC(data,days):
    N = data['Close'].diff(days)
    D = data['Close'].shift(days)
    roc = pd.Series(N/D,name='ROW')
    data = data.join(roc)
    return data 
days = 5
data = ROC(data,days)

def rescale(data):
    data = data.dropna().astype('float')
    data = sklearn.preprocessing.scale(data)
    data = pd.DataFrame(data, columns=data.columns)
    return data

def class_balance(train):
    count_class_0, count_class_1 = train['target'].value_counts()
    train_class_0 = train[train['target'] == 0]
    train_class_1 = train[train['target'] == 1]

    if count_class_0>count_class_1:
        train_class_0_under = train_class_0.sample(count_class_1)
        train_sampled = pd.concat([train_class_0_under, train_class_1], axis=0)
    else:
        train_class_1_under = train_class_1.sample(count_class_0)
        train_sampled = pd.concat([train_class_0, train_class_1_under], axis=0)
    
    print(train_sampled['target'].value_counts())
    train_sampled['target'].value_counts().plot(kind='bar', title='Count (target)')
    plt.show()
	return train_sampled
