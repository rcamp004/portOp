# -*- coding: utf-8 -*-
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
end = datetime.datetime(2015, 9, 29)

stocks = ['AAPL','BAC','BA','LMT','GLD','SBUX','TSLA','GS','DIS','GOOG']

class portfolio(object):
    def __init__(self,stocks,start,end):
        self.stocks = stocks
        self.start = start
        self.end = end
        
    def getData(self):
        data = {}
        for stock in self.stocks:
            data[stock] = web.DataReader(stock,'yahoo', start,end)
            print stock + ' sucessfully downloaded and appended.'
        pan = pd.Panel(data)
        df = pan.minor_xs('Adj Close')
        df.fillna(0,inplace = True)
        return df
        
    def returns(self,data):
        for stock in data:
            returns = data.shift(1)/data - 1    
        returns.fillna(0, inplace = True)        
        return returns
    
    def weights(self,n):
        ''' Produces n random weights that sum to 1 '''
        k = np.random.rand(n)
        return k / sum(k)
        
    def average(self,data, weights):
        'Input a data frame of daily returns and output expected returns for each stock'
        expectedReturn = data.apply(np.mean)
        expectedReturn = np.asmatrix(expectedReturn)
        mu = weights*expectedReturn.T
        return mu
    
    def covariance(self,data):
        tol = 1e-4 #Set the tolerance for covariances small enough to be set to zero.
        'Input a data frame of daily returns and output covariance matrix'
        covarianceMatrix = data.cov()
        covarianceMatrix[np.abs(covarianceMatrix) < tol] = 0           
        return covarianceMatrix
    
    def correlation(self,data):
        tol = 1e-4 #Set the tolerance for covariances small enough to be set to zero.
        cor = data.corr()
        cor[np.abs(cor) < tol] = 0           
        return cor
        
    def risk(covarmat,weights):
            sigma = np.sqrt(weights * covarmat * weights.T)
            return sigma
        
my_port = portfolio(stocks,start,end)
print my_port.stocks
print my_port.start
print my_port.end
print my_port.getData()
returnsData = my_port.getData()
print my_port.returns(returnsData)
print my_port.covariance(returnsData)
covariance = my_port.covariance(returnsData)
cor = my_port.correlation(covariance)
mu = my_port.average(returnsData,[0.1]*10)
print cor 
print mu

