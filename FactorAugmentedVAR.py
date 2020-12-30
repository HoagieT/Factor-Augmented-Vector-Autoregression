# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:00:48 2020

@author: Hogan Tong
"""

from Functions import *
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import tkinter.filedialog
from datetime import timedelta
import math 
import time
import calendar
from numba import jit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import datetime,timedelta
from statsmodels.tsa.stattools import adfuller
from numpy.linalg import cholesky

def Factor(Y, X_slow, n_factors, X_fast='None'):
    #n_time = len(Y.index)
    n_var = len(Y.columns)
    if isinstance(X_fast,str)==True:
        hist = Y.join(X_slow)
    else:
        hist = Y.join(X_slow).join(X_fast)
    
    hist=hist.dropna(axis=0,how='any')

    "step 1 - PCA on all observable variables"
    x = np.mat(hist - hist.mean())
    z = np.mat((hist - hist.mean())/hist.std())
    D, V, S = calculate_pca(hist, n_factors + n_var)
    Psi = np.mat(np.diag(np.diag(S - V.dot(D).dot(V.T))))
    factors = V.T.dot(z.T).T
    C = pd.DataFrame(data=factors, index=hist.index, columns=['C' + str(i+1) for i in range(n_factors+n_var)])
    Loadings_C = calculate_factor_loadings(hist, C)
    
    "step 2 - PCA on slow moving variables"
    x = np.mat(X_slow-X_slow.mean())
    z = np.mat((X_slow-X_slow.mean())/X_slow.std())
    D, V, S = calculate_pca(X_slow, n_factors)
    Psi = np.mat(np.diag(np.diag(S - V.dot(D).dot(V.T))))
    factors = V.T.dot(z.T).T
    F_minus = pd.DataFrame(data=factors, index=X_slow.index, columns=['F_minus' + str(i+1) for i in range(n_factors)])
    Loadings_F_slow = calculate_factor_loadings(X_slow, F_minus)
    
    "step 3 - C_t = b1*Y_t + b2*F_t"
    X = Y.join(F_minus)
    B = calculate_factor_loadings(C, X)
    Lambda_y, Lambda_f = B[:,0:n_var], B[:,n_var:]
    # F_t= Lambda_f^-1*(C_t-Lambda_y*Y_t)
    F = Lambda_f.I.dot((np.mat(C).T - Lambda_y.dot(Y.T))).T
    F = pd.DataFrame(data=F, index=X_slow.index, columns=['F' + str(i+1) for i in range(n_factors)])

    return FactorResultsWrapper(C=C, Lambda_c=Loadings_C, F_minus=F_minus, F=F)
    
class FactorResultsWrapper():
    def __init__(self, C, Lambda_c, F_minus, F):
        self.C = C
        self.Lambda_c = Lambda_c
        self.F_minus = F_minus
        self.F = F

def FAVAR(Factor, Y, lag):
    hist = Y.join(Factor)
    model=VAR(hist,missing='drop').fit(lag,trend='nc')
 
    return FAVARResultsWrapper(VAR=model)
    
class FAVARResultsWrapper():
    def __init__(self, VAR):
        self.VAR = VAR
    
    def summary(self):
        print(self.VAR.summary())
        return
    
    def predict(self, Factor, Y, step, freq='M', alpha=0.05):
        hist = Y.join(Factor)
        [forecast_mean,forecast_low,forecast_up] = self.VAR.forecast_interval(hist.values, step, alpha)
        mean = np.concatenate((hist.values, forecast_mean), axis=0)
        up = np.concatenate((hist.values, forecast_up), axis=0)
        low = np.concatenate((hist.values, forecast_low), axis=0)
        
        
        dates = pd.date_range(Y.index[0], periods=len(Y.index)+step,freq=freq)
        
        mean = pd.DataFrame(data=mean[:,0:len(Y.columns)], columns=Y.columns.tolist(), index=dates)
        low = pd.DataFrame(data=low[:,0:len(Y.columns)], columns=Y.columns.tolist(), index=dates)
        up = pd.DataFrame(data=up[:,0:len(Y.columns)], columns=Y.columns.tolist(), index=dates)
        
        return [mean,low,up]
    
    def predict_plot(self, Factor, Y, step, freq='M', alpha=0.05, figure_size=[18,12],line_width=3.0,font_size='xx-large', actural='None'):
        mean, low, up = self.predict(Factor, Y, step, freq, alpha)
        n_var = len(mean.columns)
        n_act = len(Y.index)
        
        plt.rcParams['figure.figsize'] = (figure_size[0], figure_size[1])
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['image.cmap'] = 'gray'
        
        for i in range(n_var):
            plt.figure()
            plt.plot(mean.index[n_act-1:],mean.iloc[n_act-1:,i],color='r',label='forecast', linewidth=line_width)
            plt.plot(mean.index[:n_act],mean.iloc[:n_act,i],color='k',label='observed',linewidth=line_width)
            plt.plot(mean.index[n_act-1:],low.iloc[n_act-1:,i],color='r', linestyle = '--', label='lower - '+str(int(100-alpha*100))+'%',linewidth=line_width)
            plt.plot(mean.index[n_act-1:],up.iloc[n_act-1:,i],color='r', linestyle = ':', label='upper - '+str(int(100-alpha*100))+'%',linewidth=line_width)
            plt.legend()
            if isinstance(actural,str)!=True:
                plt.plot(mean.index[n_act-1:],actural.iloc[:,i],color='k',label='observed', linewidth=line_width)
            plt.title(mean.columns[i], fontweight='bold', fontsize=font_size)
            #plt.xlabel('Date')
            #plt.ylabel('Value')
            plt.show()
        
        return
