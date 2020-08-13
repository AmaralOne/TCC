# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:17:20 2020

@author: Amaral
"""


import MethodSelector
import Ensemble_Strategist
import Predicts_Erro
import UtilsM3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ensembles:

    def __init__(self):  
       
        print("")
        self.result = {}
        self.tempoExecModelos = {}
        
        self.method_slector = MethodSelector.MethodSelector()
        self.ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
        self.predict_erros = Predicts_Erro.Predicts_Erro()
             
        
    def Ensembles_predict(self,ensembles, methods, data,tain_data,test_data):
            
        for m in methods:
            print(m)
            ts = data.copy()
            t1 = time.time()
            preditc_train, preditc_test = self.method_slector.method_Predict(m,ts,tain_data,test_data)
            tempoExec = time.time() - t1
            self.tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            self.result[m] = preditc_test
        
        print(self.result)
        forecast_errors_mse = self.predict_erros.error_MSE(methods,self.result,test_data)      
        forecast_errors_rmse = self.predict_erros.error_RMSE(methods,self.result,test_data)
        
        
        result_comb_media = self.ensembles_strategist.Mean_Combination(data,len(test_data),methods,self.result)
        result_comb_mediana = self.ensembles_strategist.Median_Combination(data,len(test_data),methods,self.result)
        result_comb_media_aparada = self.ensembles_strategist.Trimmed_Mean_Combination(data,len(test_data),methods,self.result)
        result_comb_media_ponderada = self.ensembles_strategist.weighted_average_Combination(data,len(test_data),methods,self.result,forecast_errors_mse)
        print("")
        print("erros:")
        print(forecast_errors_rmse)
        
        
        plt.plot(data,label='Original')
        plt.plot(result_comb_media, label='Comb Media')
        plt.plot(result_comb_mediana, label='Comb Mediana')   
        plt.plot(result_comb_media_aparada, label='Comb Media Aparada')   
        plt.plot(result_comb_media_ponderada, label='Comb Media Ponderada')   
        plt.legend(loc="upper left")
        plt.gcf().autofmt_xdate()
        plt.show()
        
        return self.result
        

            
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")

tamanho_teste = 18
    
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

modelos = ['ses','naive','holt','Ar','Croston', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'MLP A4','MLP A5', 'MLP A6','RNN A1','RNN A2','RNN A3',
               'RNN A4', 'RNN A5','RNN A6', 'ELM']
            
e = Ensembles()
results = e.Ensembles_predict('',modelos,ts,trainData,testData)