# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:55:14 2019
reference: https://www.marketcalls.in/python/implementation-kalman-filter-estimation-mean-python-using-pykalman-bokeh-nsepy.html
@author: Lauren
"""

from math import pi
import pandas as pd
from pandas import Series
from bokeh.plotting import figure, show, output_notebook
from datetime import date, datetime
from pykalman import KalmanFilter
from get_data import collect_data


df = collect_data('mongodb',start='2014-01-01',end='2018-01-01')
df =df.sort_index()
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = df['Close'].values[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

state_means,_ = kf.filter(df['Close'].values)
state_means = state_means.flatten()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(df.index,state_means,color='r', label='Kalman Filter Estimation')
plt.plot(df.index,df['Close'],color='g', label='Market Price')
plt.legend()
plt.show()
