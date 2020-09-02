# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:14:25 2020

@author: Amaral
"""

import sklearn
from util import Utils as ut

class Predicts_Erro:

    def __init__(self):     
        print("")
        self.freq = 12
    
    def error_MSE(self,modelos,results,testData):
        erros = {}
        
        for m in modelos:
            Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,results[m])
            erros[m] =  Erro_MSE_Test
            
        return erros
    
    def error_RMSE(self,modelos,results,testData):
        erros = {}
        
        for m in modelos:
            Erro_RMSE_Test = ut.rmse(testData,results[m])
            erros[m] =  Erro_RMSE_Test
            
        return erros
    
    def error_MAE(self,modelos,results,testData):
        erros = {}
        
        for m in modelos:
            Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,results[m])
            erros[m] =  Erro_MAE_Test
            
        return erros
    
    def error_SMAPE(self,modelos,results,testData):
        erros = {}
        
        for m in modelos:
            Erro_SMPAE_Test = ut.smape(testData,results[m])
            erros[m] =  Erro_SMPAE_Test
            
        return erros
    
    def error_MASE(self,modelos,results,testData,trainData):
        erros = {}
        
        for m in modelos:
            ut.mase_ML(trainData, testData, results[m], self.freq)
            Erro_MASE_Test = ut.smape(testData,results[m])
            erros[m] =  Erro_MASE_Test
            
        return erros
        
        
