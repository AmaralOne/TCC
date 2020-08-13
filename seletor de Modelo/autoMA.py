# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:45:52 2020

@author: Amaral
"""

import sklearn
from statsmodels.tsa.arima_model import ARMA
from random import random
from util import Utils as ut
import warnings
import pandas as pd
import numpy as np


# evaluate an MA 
def evaluate_ma_model(DATA, ARMA_order):
    # prepare training dataset
    train_size = int(len(DATA) * 0.80)
    train, test = DATA[0:train_size], DATA[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARMA(history, order=ARMA_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
            
    # calculate out of sample error
    
    error = ut.rmse(test, predictions)
    
    return error


def evaluate_models(dataset, intervalo = 4):
    warnings.filterwarnings("ignore")
    best_score, best_cfg = float("inf"), None
    mudou = False
    for m in range(intervalo+1):
        order = (0, m)
        try:
            rmse = evaluate_ma_model(dataset, order)
            if rmse < best_score:
                mudou = True
                best_score, best_cfg = rmse, order
           # print('MA%s RMSE=%.3f' % (order,rmse))
        except:
            continue
    #print('Best MA%s RMSE=%.3f' % (best_cfg, best_score))
    
    if(mudou == False):
        return (0, 1)
    
    return best_cfg

class autoMA:

    def __init__(self):  
        print()
        
        
    def fit(self,trainData):        
        
        self.trainData = trainData
        self.testData = []
        best_cfg = evaluate_models(trainData)
        self.Model = ARMA(self.trainData, order=best_cfg)
        self.Model = self.Model.fit(disp=0)
        trainingFit = pd.Series(np.ceil(self.Model.fittedvalues))
             
        return trainingFit
    
    def lag(self):
        return True
             
        
    def forecasts(self,Testdata): 
        self.testData = Testdata       
        testPredictions = pd.Series(np.ceil(self.Model.predict(
                start=len(self.trainData),
                end=len(self.trainData)+len(self.testData)-1,
                dynamic=False)))
        return testPredictions




