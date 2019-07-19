# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:45:50 2019

@author: Lauren

DNN
"""

from numpy.random import seed
from get_data import collect_data,split_data, organize_data
import numpy as np
import matplotlib.pyplot as plt

seed(1)

def create_model(u1,u2,dropout_rate,learn_rate):
        classifier = Sequential()
    	# first hidden layer
        classifier.add(Dense(units = u1, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
        #Kernel_initializer: defines the starting values for the weights of the different neurons. uniform distribution.
        #Activation: activation function for the neurons. rectified Linear Unit function or ‘relu’.
        #Input_dim:  the number of inputs to the first hidden layer
        
        # second hidden layer
        classifier.add(Dense(units = u2, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
        classifier.add(Dropout(dropout_rate))
        # output layer. require a single output, so units=1. 
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
        # the activation function is Sigmoid function because we would want to predict probability of market moving upwards.
        #classifier.add(Dropout(0.2))
        
        # compile the classifier by passing the following arguments:
        # Metrics: metrics to be evaluated by the model during the testing and training phase.
        #  We have chosen accuracy as our evaluation metric.
        optimizer = Adam(lr=learn_rate)
        classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
        # loss: https://keras.io/losses/
        # Binary Classification Problem, loss = binary_crossentropy
    
        # optimizer: https://keras.io/optimizers/
        return classifier


    
data= collect_data('csv','AAPL',start='2014-01-01',end='2019-01-01')
# remove seasonal
from statsmodels.tsa.seasonal import seasonal_decompose        
result = seasonal_decompose(data, model="Additive",freq=252,two_sided=False)# how to decide extrapolate_trend? 
result.plot()
plt.show()
trend = result.trend
seasonal = result.seasonal

data_nonseason = data-seasonal
X,Y = organize_data(data_nonseason,algoType="regression")
X_train, X_validate, X_test, Y_train, Y_validate, Y_test = split_data(X,Y)

# build Artificial Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

# =============================================================================
# def hyperparameter(X,Y):
# # tunning hyperparameter
#     #batch_size = []#[10, 20, 40, 60, 80, 100]
#     #epochs = []#range(10,100,40)
#     u1=range(20,300,20)
#     u2=range(20,300,20)
#     learn_rate = [ 0.01, 0.1, 0.2, 0.3]
#     dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     param_grid = dict( u1=u1, u2=u2,learn_rate=learn_rate,dropout_rate=dropout_rate)
#     # create model
#     from keras.wrappers.scikit_learn import KerasClassifier
#     model = KerasClassifier(build_fn=create_model, verbose=0)
#     
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.svm import SVR
#     svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                "gamma": np.logspace(-2, 2, 5)})
#     
#     grid = GridSearchCV(estimator=svr, param_grid=param_grid, n_jobs=1, cv=3)
#     # n_jobs: Number of jobs to run in parallel. -1 means using all processors
#     # cv: Determines the cross-validation splitting strategy. 
#     
#     grid_result = grid.fit(X, Y)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#     return grid_result
# 
# grid_result = hyperparameter(X,Y)
# classifier = Sequential()
# classifier.add(Dense(units = grid_result.best_params_['u1'], kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
# classifier.add(Dense(units = grid_result.best_params_['u2'], kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
# classifier.add(Dropout(grid_result.best_params_['dropout_rate']))
# classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
# optimizer = Adam(lr=grid_result.best_params_['learn_rate'])
# classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
# classifier.fit(X_train, Y_train, batch_size = grid_result.best_params_['batch_size'])
# =============================================================================
# batch siz: the number of data points the model uses to compute the error before backpropagating the errors and making modifications to the weights.
# The number of epochs: the number of times the training of the model will be performed on the train dataset.
# batch_size:Number of samples per gradient update. If unspecified, batch_size will default to 32.
# epochs: The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.

classifier = Sequential()
classifier.add(Dense(units = 400, kernel_initializer = 'normal', activation = 'tanh', input_dim = X.shape[1]))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 400, kernel_initializer = 'normal', activation = 'tanh'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation = "linear" ))
# activation: 
optimizer = Adam(lr=0.0001)
classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
history = classifier.fit(X_train, Y_train, verbose=1)
print(history.history)

# predict
y_pred = classifier.predict(X,verbose=1)

X['y_pred_return'] = y_pred 
data = data[len(data)-len(X):]
X['y_pred_price'] = data['Adj Close'] + X['y_pred_return']
#X['y_pred_price'] += seasonal['Adj Close'][len(data)-len(X):].shift(-1)
X['y_pred_indicator'] = np.where(X['y_pred_price']-data['Adj Close'] >0, 1,-1)
# compute strategy returns
#trade_dataset['Returns'] = 0.
#X['Returns'] = np.log(X['Adj Close']/X['Adj Close'].shift(1))
X['Returns'] = data['Adj Close'].shift(-1) - data['Adj Close']
#trade_dataset['Returns'] = trade_dataset['Returns'].shift(-1)

#trade_dataset['Strategy Returns'] = 0.
X['Strategy Returns']= np.where(X['y_pred_indicator'] >0, X['Returns'], -X['Returns'])

x_plot = X[len(X_train):]
x_plot['Cumulative Market Returns'] = np.cumsum(x_plot['Returns'])
x_plot['Cumulative Strategy Returns'] = np.cumsum(x_plot['Strategy Returns'])
X['Cumulative Market Returns'] = np.cumsum(X['Returns'])
X['Cumulative Strategy Returns'] = np.cumsum(X['Strategy Returns'])

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

plt.plot(X['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(X['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.axvline(x=X[len(X_train)-1:len(X_train)].index,color='k', linestyle='--')
plt.axvline(x=X[len(X_train)+len(X_validate)-1:len(X_train)+len(X_validate)].index,color='k', linestyle='--')
plt.legend()
plt.show()

