"""
Created on Mon Mar 11 15:42:10 2019
https://github.com/sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm/blob/master/knnAlgorithm.py
https://www.quantinsti.com/blog/machine-learning-k-nearest-neighbors-knn-algorithm-python
@author: Lauren

KNN
"""

import pandas as pd
import matplotlib.pyplot as plt   # Import matplotlib
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.metrics import accuracy_score
import numpy as np
import get_data

#Data cleaning
#dataset.isna().any()

# %% prepare data
data= get_data.collect_data('csv','SPY',start='2010-07-24',end='2019-01-07')
data = get_data.getTechnicalInd(data)

X,Y = get_data.get_xy(data,algoType="classification")
X = get_data.norm_data(X)
#%% build model using hyperparameter
for i in range(30,100,10):
    Xtmp = get_data.filter_feature(X,Y,i)
    print("for i as %s -------------"%i )
    Xtmp = get_data.RFE(Xtmp,Y, i)
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = get_data.split_data(Xtmp,Y)
# feature selection: remove high covariance features
#  choosing parameters for knn
    k = range(3,400,2)
    #for weight in ['uniform','distance']:
    weight='distance'# by default weights='uniform'
     #   for p1 in [1,2]: p=1 means manhattan distance. p=2 means Euclidean distance. by default p=2
    p1 = 2
    accuracy_train = accuracy_validate={}
    for n_neighbors in k:            
        knn = KNeighborsClassifier(n_neighbors ,weights = weight,p=p1, algorithm = 'auto',n_jobs=-1)#default p= 2, weights = "uniform"
        # weights = uniform, distance
        #fit model
        knn.fit(X_train, Y_train)              
        # accuracy score
        accuracy_train_curr = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_train[weight,p1,n_neighbors]=accuracy_train_curr
        accuracy_validate_curr = accuracy_score(Y_validate, knn.predict(X_validate))
        accuracy_validate[weight,p1,n_neighbors]=accuracy_validate_curr   
    # Plot the elbow
    lists = (accuracy_train.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    x = ([ele[2] for ele in x])      
    plt.plot(x,y, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('the optimal k with weights=%s, distance = %s'%(weight,p1))
    plt.show()
    plt.close()

    #accuracy_test = accuracy_score(y_test, knn.predict(X_test))
    #print("n_neighbor is %s"%n_neighbors)
    #print('train_data accuracy: %.2f' %accuracy_train_curr)
    #print('test_data accuracy: %.2f' %accuracy_test_curr)
n_neighbors = -1

for key, val in accuracy_train.items():
    if val==max(accuracy_validate.values()):
        print("optimal n_neighbor is: %s, accuracy of validate is: %.2f"%(key,val))
        weight,p1,n_neighbors= key
#%% fitting knn model
knn = KNeighborsClassifier(n_neighbors, weights=weight,p=p1, algorithm = 'auto',n_jobs=-1)
# weight function used in prediction. Possible values:
#‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
#[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

#fit model
knn.fit(X_train, Y_train)
# accuracy score
accuracy_train_curr = accuracy_score(Y_train, knn.predict(X_train))
accuracy_train[n_neighbors]=accuracy_train_curr
accuracy_validate_curr = accuracy_score(Y_validate, knn.predict(X_validate))
accuracy_validate[n_neighbors]=accuracy_validate_curr
print(accuracy_train_curr,accuracy_validate_curr)    
# predict buy/sell signal
#%% prediction
prob = knn.predict_proba(X)

X['predicted'] = knn.predict(X)



#%% plot returns 
data['returns'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
data = data[len(data)-len(X):]
X['returns'] = data['returns'][len(data)-len(X):]
split = len(X_train)+len(X_validate)
split=0
cummulative_returns = data[split:]['returns'].cumsum()*100
X['strategy_returns'] = X['predicted'].shift(1)*X['returns']
cum_strategy_returns = X[split:]['strategy_returns'].cumsum()*100

plt.figure(figsize=(10,5))
plt.plot(cummulative_returns,label = 'Empirical Returns')
plt.plot(cum_strategy_returns, label = "Strategy Returns")
plt.axvline(x=X[len(X_train)-1:len(X_train)].index,color='k', linestyle='--')
plt.axvline(x=X[len(X_train)+len(X_validate)-1:len(X_train)+len(X_validate)].index,color='k', linestyle='--')
plt.title("Return for validation set")
plt.legend()
plt.show()

#%% cross validation to check if model is stable
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold)
print("Train score :", results)
print("Train score - mean: %2.1f"% (results.mean()*100),"%")
print("Train score - std:%2.1f"% (np.std(results)*100),"%") 
