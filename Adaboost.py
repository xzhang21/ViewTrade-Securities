# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:44:23 2019

@author: Lauren

DNN + Adaboost
"""
from get_data import collect_data,split_data, organize_data
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
import numpy as np
#import talib
#series = Series.from_csv('XOM.csv', header=0)# if header=0, skip 0 row
data= collect_data('yahoo',start='2014-01-01',end='2019-01-01')


from statsmodels.tsa.seasonal import seasonal_decompose        
result = seasonal_decompose(data, model="Additive",freq=252,two_sided=False)# how to decide extrapolate_trend? 
result.plot()
plt.show()
trend = result.trend
seasonal = result.seasonal

data = data-seasonal
X,Y = organize_data(data,"regression")
X_train, X_test, X_validate, Y_train, Y_test, Y_validate = split_data(X,Y)


from keras.models import Sequential
from keras.layers import Dense
def create_model():
    classifier = Sequential()
    classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
    classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return classifier

# create model
from keras.wrappers.scikit_learn import KerasClassifier
# create model
model = KerasClassifier(build_fn=create_model,verbose=2,batch_size=10)
# verbose shows training progress
from sklearn.ensemble import AdaBoostClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=3,base_estimator=model, learning_rate=1)
# base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
# n_estimators: Number of weak learners to train iteratively.
# learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.
# Import Support Vector Classifier
#from sklearn.svm import SVC
#svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
#abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)


# Train Adaboost Classifer
model = abc.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = model.predict(X)
X['y_pred_return'] = y_pred
X['y_pred_price'] = X['Adj Close'].shift(1) + X['y_pred_return'].shift(1)
X['y_pred_price'] += seasonal['Adj Close'][len(data)-len(X):]
X['y_pred_indicator'] = np.where(X['y_pred_price'].shift(-1)-X['Adj Close'] >0, 1,-1)

# compute strategy returns
#trade_dataset['Returns'] = 0.
#X['Returns'] = np.log(X['Adj Close']/X['Adj Close'].shift(1))
X['Returns'] = X['Adj Close'] - X['Adj Close'].shift(1)
#trade_dataset['Returns'] = trade_dataset['Returns'].shift(-1)

#trade_dataset['Strategy Returns'] = 0.
X['Strategy Returns'] = np.where(X['y_pred_indicator'] >0, X['Returns'], - X['Returns'])

x_plot = X[len(X_train):]
x_plot['Cumulative Market Returns'] = np.cumsum(x_plot['Returns'])
x_plot['Cumulative Strategy Returns'] = np.cumsum(x_plot['Strategy Returns'])
X['Cumulative Market Returns'] = np.cumsum(X['Returns'])
X['Cumulative Strategy Returns'] = np.cumsum(X['Strategy Returns'])

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(x_plot['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(x_plot['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
data['y_pred'] = np.NaN
data.iloc[(len(data) - len(y_pred)):,-1:] = y_pred
trade_dataset = data.dropna()# keep only test data

# compute strategy returns
#trade_dataset['Returns'] = 0.
trade_dataset['Returns'] = np.log(trade_dataset['Adj Close']/trade_dataset['Adj Close'].shift(1))
#trade_dataset['Returns'] = trade_dataset['Returns'].shift(-1)

#trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] >0, trade_dataset['Returns'], -trade_dataset['Returns'])


trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

