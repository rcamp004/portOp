"""
Spyder Editor

A Markowitz based Portfolio Optimzer
@author: Richard Campanha
"""



from pandas_datareader import data as web
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import matplotlib
import cvxopt as opt
import pandas as pd
import numpy as np
import datetime

np.random.seed(123)


#Year, Month, Day
start = datetime.datetime(2008, 10, 1)
end = datetime.datetime(2015, 9, 24)

stocks = ['AAPL','BAC','BA','LMT','GLD','SBUX','TSLA','GS','DIS','GOOG']
insBook = ['AIZ','AXS','SLV','PRU','TRV','VR']
frakkers = ['ABX','PALL','PPLT','COPX','PGM','LQD','SLV','GLD','SWC','PTM','AGG','VCSH','BND','AUY']

port = stocks + insBook + frakkers
port1 = stocks + insBook

def getData(stocks):
    data = {}
    for stock in stocks:
        data[stock] = web.DataReader(stock,'yahoo', start,end)
        print stock + ' sucessfully downloaded and appended.'
    pan = pd.Panel(data)
    df = pan.minor_xs('Adj Close')
    df.fillna(0,inplace = True)
    return df

def calcReturns(returnData):
    for stock in returnData:
        returns = returnData.shift(1)/returnData - 1    
    returns.fillna(0, inplace = True)        
    return returns
    
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
    
def average(returnData, weights):
    'Input a data frame of daily returns and output expected returns for each stock'
    expectedReturn = returnData.apply(np.mean)
    expectedReturn = np.asmatrix(expectedReturn)
    mu = weights*expectedReturn.T
    return mu

def cvmat(returnData):
    tol = 1e-4 #Set the tolerance for covariances small enough to be set to zero.
    'Input a data frame of daily returns and output covariance matrix'
    covarianceMatrix = returnData.cov()
    covarianceMatrix[np.abs(covarianceMatrix) < tol] = 0           
    return covarianceMatrix
    
def risk(covarmat,weights):
        sigma = np.sqrt(weights * covarmat * weights.T)
        return sigma
    
tol = 1e-4
def portfolio(returns,w):
    p = returns.apply(np.mean)
    mu = w * p.T
    c = returns.cov()
    c[np.abs(c) < tol] = 0
    sigma = np.sqrt(w * c * w.T)
    
    return mu,sigma
    
  
#closeData = getData(stocks)
#closeData = getData(insBook)
closeData = getData(port)

#weights = rand_weights(len(stocks))
weights = rand_weights(len(port))
dailyReturns = calcReturns(closeData)
expectedReturn = average(dailyReturns,weights)
covarianceMatrix = cvmat(dailyReturns)
sigma = risk(covarianceMatrix,weights)