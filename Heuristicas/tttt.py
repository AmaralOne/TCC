# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:25:45 2020

@author: Amaral
"""

import pandas as pd
import Ensemble_Strategist
import UtilsCIF
import matplotlib.pyplot as plt
import Ensemble_Strategist
import Predicts_Erro
import sklearn
from util import Utils as ut

ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()

cif = UtilsCIF.UtilsCIF()
serie = 'ts46'
arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_validacao30Porcent_'+serie+'.xlsx',None)
#arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_'+serie+'.xlsx',None)
            
result_validation = arquivo_result.pop('predict_validation')

#configuração 7
modelos_serieTs66 = ['NNAR', 'NNAR RNN', 'SVR A6']        #médiana
modelos_serieTs69 = ['holt', 'SVR A6']        #média ponderada
modelos_serieTs46 = ['holt', 'Arima','NNAR']        #médiana
modelos_serieTs56 = ['naive', 'NNAR RNN']        #média ponderada
modelos_serieTs63 = ['holt','Ar', 'NNAR RNN','SVR A2']        #média ponderada

#configuração 19
modelos_serieTs66 = ['NNAR', 'NNAR RNN', 'SVR A6']        #média Ponderada
modelos_serieTs69 = ['SVR A1',	'SVR A2',	'ELM']        #mediana
modelos_serieTs46 = ['naive', 'SVR A6']        #média ponderada
modelos_serieTs56 = ['ses',	'holt',	'MLP A1']        #média ponderada
modelos_serieTs63 = ['ses',	'naive',	'NNAR',	'RNN A1',	'ELM']        #média aparda

#configuração 20
modelos_serieTs66 = ['ses'	,'naive'	,'Ar','SVR A2','SVR A3','SVR A6','NNAR RNN']  #mediana
modelos_serieTs69 = ['holt', 'SVR A6']        #média ponderada
modelos_serieTs46 = ['holt','Arima','SVR A6','NNAR RNN','RNN A1','RNN A2']    #média
modelos_serieTs56 = ['NNAR','SVR A1']        #média ponderada
modelos_serieTs63 = ['Arima','SVR A1']        #média ponderada

modelos = modelos_serieTs46      
estrategia_combination = 'media'  
#estrategia_combination = 'mediana'
#estrategia_combination = 'media aparada'
#estrategia_combination = 'media ponderada'



results = {}
for m in range(len(result_validation)):
    if result_validation.iloc[m][0] not in modelos:
        continue           
    results[result_validation.iloc[m][0]] = pd.Series(result_validation.iloc[m][1:])
    
ts, tamanho_teste = cif.serie(serie)
                
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
tamanho_validacao = (int)((len(ts)-tamanho_teste)*0.3)
inico_de_validacao = (tamanho_serie-(tamanho_teste+tamanho_validacao))
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]
validationData = ts[inico_de_validacao:incio_de_teste]

result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif_Retreino\\Resultado_Predict_retreino_'+serie+'.xlsx',None)   
result_prediction = result_series.pop('predict_test')
            
ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
predict_erros = Predicts_Erro.Predicts_Erro()
results_t = {}
for m in range(len(result_prediction)):
    if result_prediction.iloc[m][0] not in modelos:
        continue           
    results_t[result_prediction.iloc[m][0]] = pd.Series(result_prediction.iloc[m][1:])
    
modelos_escolhidos = modelos[:] 
plt.plot(ts)

for m in modelos_escolhidos:
    t = pd.Series(results_t[m].values,ts[incio_de_teste:].index)
    plt.plot(t, label=m+ "_test")
    t_V = pd.Series(results[m].values,ts[inico_de_validacao:incio_de_teste].index)
    plt.plot(t_V, label=m+ "_validation")
    print(m+": "+str(ut.smape(testData,t)))


if estrategia_combination == 'media':
    result_comb_v = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
    result_comb = ensembles_strategist.Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
    print("Media")

elif estrategia_combination == 'mediana':
    result_comb_v = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
    result_comb = ensembles_strategist.Median_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)       
    print("Mediana")
    
elif estrategia_combination == 'media aparada':
    result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
    result_comb_v = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
    print("Média aparada")   

elif estrategia_combination == 'media ponderada':
    forecast_errors_mse = predict_erros.error_RMSE(modelos_escolhidos,results,validationData) 
    result_comb = ensembles_strategist.weighted_average_Combination(ts,tamanho_teste,modelos_escolhidos,results_t,forecast_errors_mse)    
    result_comb_v = ensembles_strategist.weighted_average_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results,forecast_errors_mse)
    print("Média Ponderada")



plt.plot(result_comb_v,label="Comb AG "+estrategia_combination)
plt.plot(result_comb,label="Comb AG "+estrategia_combination)
plt.title("Resultado AG Ensemble Selection "+serie)
plt.legend(loc="upper left")
plt.gcf().autofmt_xdate()
plt.show()

print("AG: "+str(ut.smape(testData,result_comb)))


