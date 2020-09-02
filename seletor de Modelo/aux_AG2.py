# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:52:54 2020

@author: Amaral


"""

import UtilsM3
import matplotlib.pyplot as plt
import Ensemble_Strategist
import Predicts_Erro
import random
import pandas as pd

arquivo_result = pd.read_excel("Resultado_Predict_N1679.xlsx")

results = {}
for m in range(len(arquivo_result)):
    print(arquivo_result.iloc[m][0])
    results[arquivo_result.iloc[m][0]] = pd.Series(arquivo_result.iloc[m][1:])


U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")

tamanho_teste = 18
    
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]





methods = ['ses','naive','holt','Ar','Croston', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'MLP A4','MLP A5', 'MLP A6','RNN A1','RNN A2','RNN A3',
               'RNN A4', 'RNN A5','RNN A6', 'ELM']
ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
predict_erros = Predicts_Erro.Predicts_Erro()

result_comb_media = ensembles_strategist.Mean_Combination(ts,len(testData),methods,results)

plt.plot(ts,label='Original')
plt.plot(result_comb_media, label='Comb Media')
plt.legend(loc="upper left")
plt.gcf().autofmt_xdate()
plt.show()

forecast_errors_mse = predict_erros.error_MSE(methods,results,testData)   

cromossomo = []
        
for i in range(len(methods)+2):
    if random.random() < 0.5:
        cromossomo.append("0")
    else:
        cromossomo.append("1")
        
        
modelos = ['ses','naive','holt','Ar','Croston', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'MLP A4','MLP A5', 'MLP A6','RNN A1','RNN A2','RNN A3',
               'RNN A4', 'RNN A5','RNN A6', 'ELM']

import sklearn
        
def avaliacao():
        print(cromossomo)
        erro = 1000000
        methods_selecionados = []
        tamanho_cromossomo = len(cromossomo)
        for i in range(tamanho_cromossomo):
           if cromossomo[i] == '1':
               methods_selecionados.append(modelos[i])
        for i in range(4):
            print("Metodos Selecionados: ",methods_selecionados)
            if i == 0:
                print("Media")
                result_comb = ensembles_strategist.Mean_Combination(ts,len(testData),methods_selecionados,results)
            elif i == 1:
                result_comb = ensembles_strategist.Median_Combination(ts,len(testData),methods_selecionados,results)
                print("Mediana")
            elif i == 2:
                result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts,len(testData),methods_selecionados,results)
                print("Media Aparada")
            elif i == 3:
                print("Media Ponderada")
                result_comb = ensembles_strategist.weighted_average_Combination(ts,len(testData),methods_selecionados,results,forecast_errors_mse)
            
        erro = sklearn.metrics.mean_squared_error(testData,result_comb) 
        nota_avaliacao = 1/erro
        print(erro)
        print(nota_avaliacao)

avaliacao()