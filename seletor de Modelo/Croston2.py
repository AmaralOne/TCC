# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:55:19 2020

@author: Amaral
"""
import croston5 as cr
import pandas as pd
import numpy as np

class Croston2:

    def __init__(self):  
        print()
        
        
    def fit(self,trainData):        
        
        self.trainData = trainData
        self.testData = []
        self.Model = cr.Croston(self.trainData)
        self.Model.fit()
        trainingFit = pd.Series(np.ceil(self.Model.fittedForecasts))
        
        
             
        return trainingFit
    
    def lag(self):
        return False
             
        
    def forecasts(self,Testdata): 
        self.testData = Testdata       
        testPredictions = pd.Series(np.ceil(self.Model.forecast(len(self.testData))))
        return testPredictions