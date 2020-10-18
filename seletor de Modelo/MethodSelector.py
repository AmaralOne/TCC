# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:27:00 2020

@author: Amaral
"""


import warnings
import Ses
import Naive
import Holt
import AR
import Croston2 as cr
import ML_M4
import autoSVR
import ARIMA
import ML_Otexts


class MethodSelector:

    def __init__(self):  
        #Ignorar messagens de avisos de algumas bibiotecas
        warnings.filterwarnings("ignore")
        
        
        
    def method_Predict(self,method,data,tain_data,test_data, return_best = 0):
        
        
        tamanho_teste = len(test_data)
        
        if(method == 'ses'):
                       
            model = Ses.Ses()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)
                      
        elif (method == 'naive'):
                      
            model = Naive.Naive()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)
            
        elif (method == 'holt'):
            
            model = Holt.Holt()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)

        elif (method == 'Ar'):

            model = AR.AR()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)

        elif (method == 'Croston'):

            model = cr.Croston2()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)

        elif (method == 'Arima'):

            model = ARIMA.ARIMA()
            preditc_train = model.fit(tain_data)
            preditc_test = model.forecasts(test_data)
            
        elif (method == 'SVR A1'):
            
            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,1)
            preditc_train, preditc_test = model.fit()
            
        elif (method == 'SVR A2'):
            
            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,2)
            preditc_train, preditc_test = model.fit()
            
        
        elif (method == 'SVR A3'):

            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,3)
            preditc_train, preditc_test = model.fit()

        
        elif (method == 'SVR A4'):

            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,4)
            preditc_train, preditc_test = model.fit()

            
            
        elif (method == 'SVR A5'):

            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,5)
            preditc_train, preditc_test = model.fit()

            
        elif (method == 'SVR A6'): 

            model = autoSVR.autoSVR()
            model.prepararSerie(data,tamanho_teste,1,6)
            preditc_train, preditc_test = model.fit()
            
            
        elif (method == 'NNAR'):

            model = ML_Otexts.ML_Otexts()
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit(data,test_data,tamanho_teste,(len(data)-tamanho_teste),0,1,0,True,20,'sklearn')
    
            
        elif (method == 'NNAR RNN'): 

            model = ML_Otexts.ML_Otexts()
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit(data,test_data,tamanho_teste,(len(data)-tamanho_teste),0,1,0,True,20,'rnn')
             
            
        elif (method == 'MLP A1'):

            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,1)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'MLP A2'):
   
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,2)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'MLP A3'):
     
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,3)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'MLP A4'):

            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,4)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'MLP A5'):

            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,5)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'MLP A6'):
   
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,6)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_MLP()

        elif (method == 'RNN A1'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,1)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best= model.fit_RNN()
            
            
        elif (method == 'RNN A2'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,2)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_RNN()
            
        elif (method == 'RNN A3'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,3)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_RNN()
            
        elif (method == 'RNN A4'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,1,4)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_RNN()
            
        elif (method == 'RNN A5'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data[:],tamanho_teste,12,5)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_RNN()
            
        elif (method == 'RNN A6'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data,tamanho_teste,12,6)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_RNN()
            
        elif (method == 'ELM'):
            
            model = ML_M4.ML_M4()
            model.prepararSerie(data,tamanho_teste,12,1)
            preditc_train, preditc_test, preditc_train_best, preditc_test_best = model.fit_ELM()
            
        if(return_best == 0):
            return preditc_train, preditc_test
        return preditc_train, preditc_test, preditc_train_best, preditc_test_best
        