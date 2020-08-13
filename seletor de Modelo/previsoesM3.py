# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:06:11 2020

@author: Amaral
"""
import pandas as pd
from util import Utils as ut
import UtilsM3
import matplotlib.pyplot as plt
import sklearn


file = u'M3Forecast.xls'
path = 'dataset/'
freq = 'M'

arquivo = pd.read_excel(path+file,None)


serie = "N1679"

#Carregar uma Série Temporal do Conjunto M3
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame(serie)

#Imprimir Gráfico da Séreie
plt.plot(ts)
plt.title(serie)
plt.show()


tamanho_teste = 18

#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]



cols = ['Modelo','RMSE','MSE','MAE']
resultSheet1 = pd.DataFrame(columns=cols)

for k, item in arquivo.items():
    print(k)
    df = item

    aux = df[df.iloc[:,1]==tamanho_teste]
    previsao = aux[aux.iloc[:,0]==serie]
    previsao = previsao.iloc[0,2:]

    previsao = pd.Series(previsao)

    Erro_RMSE_Test = ut.rmse(testData,previsao)
    
    Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,previsao)
    Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,previsao)
    Erro_RMSE_Test = ut.rmse(testData,previsao)   
    line = {'Modelo':k,
                'RMSE': round(Erro_RMSE_Test,2),
                'MSE': round(Erro_MSE_Test,2),
                'MAE': round(Erro_MAE_Test,2)}
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    print(line)

resultSheet1.to_excel(excel_writer='Previsao'+serie+'.xlsx',index=False)