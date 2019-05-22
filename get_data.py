# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:08:48 2019

@author: Lauren
"""
import numpy as np
import pandas as pd
from talib import ROC, WILLR, RSI, ATR,OBV ,ADOSC, STOCH, STOCHF,APO


def collect_data(source, tick='SPY',start='1980-01-01',end='2019-01-01'):
    if source=="yahoo":
        import fix_yahoo_finance as yf  
        data = yf.download(tick,start,end)
    # dataBench = yf.download('SPY','2013-01-01','2018-01-01')
    # series = data['Adj Close'] - dataBench['Adj Close']

    if source == 'csv':
    #series = Series.from_csv('XOM.csv', header=0)# if header=0, skip 0 row
        data = pd.read_csv('GSPC.csv',index_col="Date",parse_dates=True)
        data = data.loc[start:end ,:]
    
    if source == 'mongodb':
        from KafkaProcess import MongoDBConnect
        mongo = MongoDBConnect(IPAddress="192.168.110.116", Port=27017, dbName="chart", Collection="Day")
        stock_high, stock_low, stock_close,stock_open, stock_volume,stock_time = mongo.mongodb_connect(tick, start.split('-'), end.split('-'))
        data = pd.DataFrame({'High':stock_high,'Low':stock_low,  'Open':stock_open, 'Close':stock_close,'Volume':stock_volume} ,index=stock_time)
        #data = pd.Series( data )
    return data



def getTechnicalInd(data, window = 10):
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']    
    #madev = roc = willr=rsi=apo=atr=obv=adosc=slowk=fastk=slowd=fastd=vpos=vneg={}
    window = 10
    for i in range(0,window):

        # momentum indicators
        # rate of change: (price/prevPrice - 1)*100
        data["roc{0}".format(i)] = ROC(close.shift(i), timeperiod=10) 
        # williams oscillator
        data["willr{0}".format(i)] = WILLR(high.shift(i), low.shift(i), close.shift(i), timeperiod=14)
        
        # relative strength index
        data["rsi{0}".format(i)] = RSI(close.shift(i), timeperiod=14)
        
        # absolute price oscillator: slowMA(price) - fastMA(price)
        data["apo{0}".format(i)] = APO(close.shift(i), fastperiod=12, slowperiod=26, matype=0)
        # volatility indicator
                          
        data["atr{0}".format(i)] = ATR(high.shift(i), low.shift(i), close.shift(i), timeperiod=14) # average true range
        # Volume indicator
        # on balance volume
        data["obv{0}".format(i)] = OBV(close.shift(i), volume.shift(i))
        # chaikin A/D oscillator
        data["adosc{0}".format(i)] = ADOSC(high.shift(i), low.shift(i), close.shift(i), volume.shift(i), fastperiod=3, slowperiod=10)
        
        # Stochastic Oscillator Slow %k: (C-L)/(H-L)*100	
        data["slowk{0}".format(i)], data["slowd{0}".format(i)] = STOCH(high.shift(i), low.shift(i), close.shift(i), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        
        # STOCHF - Stochastic Fast
        data["fastk{0}".format(i)], data["fastd{0}".format(i)] = STOCHF(high.shift(i), low.shift(i), close.shift(i), fastk_period=5, fastd_period=3, fastd_matype=0) 

        #vortex
        data["vortex_indicator_pos{0}".format(i)] = vortex_indicator_pos(high.shift(i), low.shift(i), close.shift(i), n=14, fillna=False)
        data["vortex_indicator_neg{0}".format(i)] =   vortex_indicator_neg(high.shift(i), low.shift(i), close.shift(i), n=14, fillna=False)  
               
        # returns
    for i in range(1,window):
        # overlap studies
        data["madev{0}".format(i)] = close-close.shift(1).rolling(window = i).mean()
        data["return{0}".format(i)] = close-close.shift(i)
    # std
    for i in range(2,window):
        data["std{0}".format(i)] = close.rolling(i).std() #Standard deviation for a period of 5 days
    
    return data#madev, roc,willr,rsi,apo,atr,obv,adosc,slowk,slowd,fastk,fastd,vpos, vneg, returns, stds

     
# vortex indicator
def vortex_indicator_pos(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vip = vmp.rolling(n).sum() / trn
    if fillna:
        vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vip, name='vip')


def vortex_indicator_neg(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bearish signal triggers when the negative trend indicator
    crosses above the positive trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    if fillna:
        vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vin, name='vin')

def norm_data(X):
    # standardize data. makes the mean of all the input features equal to zero and also converts their variance to 1
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    return X

   
def get_xy(data,algoType):
    data = data.dropna()
    X = data.drop(columns=['High', 'Low','Adj Close','Volume','Open','Close'])
    #X = data[list(returns.keys())]
    #X = data[[#'3day MA deviation','H-L','10day MA deviation','30day MA deviation','O-C','Std_dev lag10','Std_dev lag3','Std_dev lag30',
              #,'Std_dev lag10','Std_dev lag30'
            #'Close lag1','Close lag2','Close lag3','Close lag4','Close lag5','Close lag6','Close lag7','Close lag8',
             # 'Close lag9','Close lag10',
             #'Adj Close',
             #'Return lag1', 'Return lag2', 'Return lag3', 'Return lag4', 'Return lag5', 'Return lag6', 'Return lag7', 'Return lag8',
              #'Open lag1','Open lag2','Open lag3','Open lag4','Open lag5','Open lag6','Open lag7','Open lag8',
              #'Open lag9','Open lag10','Open',
              #'High lag1','High lag2','High lag3','High lag4','High lag5','High lag6','High lag7','High lag8',
              #'High lag9','High lag10','High',
              #'Low lag1','Low lag2','Low lag3','Low lag4','Low lag5','Low lag6','Low lag7','Low lag8',
              #'Low lag9','Low lag10','Low',
              #'Std_dev lag3','Std_dev lag10','Std_dev lag30'
              #]]
    data['Price_Rise'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)
    # the target variable is whether price will close up or down on the next trading day. 
    # if the tomorrow’s closing price is greater, then we will buy, else we will sell. 
    data['Price_Change'] = data['Adj Close'].shift(-1) - data['Adj Close']
    
    if algoType=="classification":
        Y = data['Price_Rise']# store last column
    else:
        Y = data['Price_Change']
    return X, Y


def split_data(X, Y):


    split1 = int(len(X)*0.7)
    split2 = int(len(X)*0.9)
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = X[:split1], X[split1:split2],X[split2:], Y[:split1], Y[split1:split2], Y[split2:]
    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

    
def filter_feature(X,Y,kin=100):
    # firstly, select first 50 features using chi-squared
    from sklearn.feature_selection import SelectKBest,mutual_info_classif
    #from sklearn.feature_selection import chi2
    # Feature extraction
    #mi = mutual_info_classif(X, y, discrete_features=’auto’, n_neighbors=3, copy=True, random_state=None)
    test_mi = SelectKBest(score_func = mutual_info_classif, k=kin)# select k features
    fit_mi = test_mi.fit(X,Y)
    #test_chi2 = SelectKBest(score_func=chi2, k=3)
    #fit_chi2 = test_chi2.fit(X, Y)
    
    # Summarize scores
    np.set_printoptions(precision=3)
    #print("chi2: "+fit_chi2.scores_)
    X = fit_mi.transform(X)
    # Summarize selected features
    X = pd.DataFrame(X)
    return X

def RFE(X,Y, n_features):# recursive feature selection using logistic regression    
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    # Feature extraction
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select  = n_features)# The number of features to select
    fit = rfe.fit(X, Y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    # Ridge regression to determine the coefficient R2.
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X,Y)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
    # A helper method for pretty-printing the coefficients
    print ("Ridge model:", pretty_print_coefs(ridge.coef_))
    import operator
    fitsupport = list(map(operator.not_, fit.support_))
    X = X.drop(X.columns[fitsupport],axis=1)

    return X

def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)    
    
    
    
    
