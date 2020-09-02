# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:21:47 2020

@author: Amaral
"""
import UtilsM3
import time
import numpy as np
import pandas as pd
import Ensembles
            
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")

tamanho_teste = 18
    
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'RNN A1','RNN A2','RNN A3',
                'ELM']
            
e = Ensembles()
results, tempoExecModelos = e.Ensembles_predict('',modelos,ts,trainData,testData)

results['ses'][0]

cols = ['Modelo','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','tempo']
resultSheet1 = pd.DataFrame(columns=cols)
for m in modelos:
    line = {'Modelo':m,
                    '1': results[m][0],
                    '2': results[m][1],
                    '3': results[m][2],
                    '4': results[m][3],
                    '5': results[m][4],
                    '6': results[m][5],
                    '7': results[m][6],
                    '8': results[m][7],
                    '9': results[m][8],
                    '10': results[m][9],
                    '11': results[m][10],
                    '12': results[m][11],
                    '13': results[m][12],
                    '14': results[m][13],
                    '15': results[m][14],
                    '16': results[m][15],
                    '17': results[m][16],
                    '18': results[m][17],
                    'tempo':tempoExecModelos[m]
                }
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    
resultSheet1.to_excel(excel_writer='Resultado_Predict_N1679_3.xlsx',index=False)