# -*- coding: utf-8 -*-
"""
Spyder Editor

A Markowitz based Portfolio Optimzer
@author: Richard Campanha
"""


from numpy import array, zeros, matrix, ones, shape, linspace, hstack
from pandas_datareader import data as web
from pandas.io.data import get_data_yahoo
from pandas import Series, DataFrame
#from cvxopt import blas, solvers
import matplotlib.pyplot as plt
from numpy.linalg import inv
#import cvxopt as opt
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib
import datetime
import sys

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
        
    def average(self,data, weights = 1):
        'Input a data frame of daily returns and output expected returns for each stock'
        if isinstance(data,pd.Series):  #Check if data is a vector.
            return np.average(data)
        else:
            expectedReturn = data.apply(np.mean)
            expectedReturn = np.asmatrix(expectedReturn)
            mu = weights*expectedReturn.T
            return mu
        
    def cov(self,data):
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
        
    def risk(self,covarmat,weights):
            sigma = np.sqrt(weights * covarmat * weights.T)
            return sigma

    def individual_returns(self,returns):
        expected_individual_return = {stock:myport.average(returns[stock],1) for stock in returns} #myport needs to be more general and not Global!!!
        return expected_individual_return
        
    def max_ind_ret(self,ind_ret_dict):
        #Largest single return from stock list: 
        largest_return_stock = max(ind_ret_dict, key = ind_ret_dict.get) #get Key
        max_val = ind_ret_dict.get(largest_return_stock) #print value
        return(largest_return_stock,max_val)
        
    def graphStock(self,stockData):
        stockData.plot(subplots = True, legend = True,figsize = (8, 8));        
        plt.show()
        
    def presentValue(cashflow, inrate):
        k = len(cashflow)
        pv = 0
        if isinstance(inrate,list):
            if len(inrate) != len(cashflow):
                print "Interest rate array needs to be same length as cashflow array"            
            else:
                for payout in range(k):
                    pv = pv + cashflow[payout] / ((1+inrate[payout])**payout)
                    print pv
                return pv    
        elif isinstance(inrate,float):
            for payout in range(k):                
                pv = pv + cashflow[payout] / ((1+inrate)**payout)
                print pv
            return pv    

    def futureValue(cashflow,inrate):
        k = len(cashflow)
        fv = 0
        if isinstance(inrate,list):
           if len(inrate) != len(cashflow):
               print "Interest rate array needs to be same length as cashflow array"            
           else:
               for payout in range(k):
                   fv = fv + cashflow[payout]*(1+inrate[payout])**(k-payout-1)
               return fv
        elif isinstance(inrate,float):
            for payout in range(k):
                fv = fv + cashflow[payout]*(1+inrate)**(k-payout-1)
#                print "%s to the %s power" % (payout,k-payout)
#                print fv
            return fv    
        
    def pvPerpetuity(val,inrate):
        return val/inrate



#print individual stock returns
myport = portfolio(stocks,start,end)
stockData = myport.getData()
returnsData = myport.returns(stockData)
indret = myport.individual_returns(returnsData)
print returnsData
print indret
print myport.max_ind_ret(indret)

#Test Function Calls            
print myport.stocks
print myport.start
print myport.end
print myport.getData()
print myport.returns(stockData)
print myport.cov(returnsData)
covariance = myport.cov(returnsData)
cor = myport.correlation(covariance)
mu = myport.average(returnsData,[0.1]*10)
print cor 
print mu
myport.graphStock(stockData[['AAPL','TSLA','GLD']])
#myport.efficient_frontier_plot()
