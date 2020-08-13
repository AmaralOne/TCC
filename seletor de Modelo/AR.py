# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:55:19 2020

@author: Amaral
"""
import statsmodels.tsa.ar_model as ar
import pandas as pd
import numpy as np

class AR:

    def __init__(self):  
        print()
        
        
    def fit(self,trainData):        
        
        self.trainData = trainData
        self.testData = []
        self.Model = ar.AR(trainData)
        self.Model = self.Model.fit()
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