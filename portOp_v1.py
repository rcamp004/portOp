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
end = datetime.datetime(2015, 9, 24)

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
        
my_port = portfolio(stocks,start,end)
print my_port.stocks
print my_port.start
print my_port.end
print my_port.getData()
