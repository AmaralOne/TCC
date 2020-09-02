# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:55:19 2020

@author: Amaral
"""
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy as np

class ARIMA:

    def __init__(self):  
        print()
        
        
    def fit(self,trainData,freq=12):        
        
        self.trainData = trainData
        self.testData = []
        
        # Fit a simple auto_arima model
        
        try:
            trainingFit= self.Model.predict_in_sample(exogenous=None, return_conf_int=False)
        except:
            self.Model = pm.auto_arima(self.trainData, error_action='ignore', trace=False,
                      suppress_warnings=True, maxiter=20,
                      seasonal=True, m=1, return_valid_fits = False )
            
            trainingFit= self.Model.predict_in_sample(exogenous=None, return_conf_int=False)
        
        
        trainPredictions = pd.Series(trainingFit,self.trainData.index)
        
        #print(self.Model.summary())
        
        return trainPredictions
    
    def lag(self):
        return True
             
        
    def forecasts(self,Testdata): 
        self.testData = Testdata       
        testPredictions = self.Model.predict(self.testData.shape[0]) 
        

        testPredictions = pd.Series(testPredictions,self.testData.index)
        return testPredictions