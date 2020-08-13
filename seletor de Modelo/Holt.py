# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:55:19 2020

@author: Amaral
"""
import statsmodels.tsa.holtwinters as ts
import pandas as pd
import numpy as np

class Holt:

    def __init__(self):  
        print()
        
        
    def fit(self,trainData):
        
        
        self.Model = ts.Holt(trainData)
        self.Model = self.Model.fit(optimized = True,
                                                   use_brute = True) #grid search
        
        trainingFit = pd.Series(np.ceil(self.Model.fittedvalues))
             
        return trainingFit
             
        
    def forecasts(self,Testdata):        
        testPredictions = pd.Series(np.ceil(self.Model.forecast(len(Testdata))))
        return testPredictions
    
    def lag(self):
        return False