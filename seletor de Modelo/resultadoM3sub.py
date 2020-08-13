# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:03:00 2020

@author: Amaral
"""

import UtilsM3
import pandas as pd
import numpy as np

base_dados = 'sub_M3'
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()

cols = ['Modelo','SMAPE', 'SMAPE (STD)','MASE', 'MASE (STD)','RMSE', 'RMSE (STD)','MSE','MSE (STD)','MAE','MAE (STD)','Tempo']
resultSheetTotal = pd.DataFrame(columns=cols)

smape = []
mase = []
rmse = []
mse = []
mae = []
tempos = []

modelos = ['ses','naive','holt','Ar','Croston', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'MLP A4','MLP A5', 'MLP A6','RNN A1','RNN A2','RNN A3',
               'RNN A4', 'RNN A5','RNN A6', 'ELM']
modelos.append('Comb Mediana')
modelos.append('Comb Média')
modelos.append('Comb Média Ponderada')

for serie in index:
    ts = U_m3.buildM3DataFrame(serie)
    
    if(not(len(ts)>= 99) ):
        continue
    
    print('Série '+serie)
    
    aux = 'Resultado_'+base_dados+'_'+serie+'.xlsx'
    aqruivo = pd.read_excel(aux,None)
    aqruivo = aqruivo.pop('Sheet1')
    
    print(aux)
    
    temp = aqruivo['SMAPE'][:len(aqruivo['SMAPE'])-3].values
    temp = temp.astype(float)
    smape.append(temp)
    
    temp = aqruivo['MASE'][:len(aqruivo['MASE'])-3].values
    temp = temp.astype(float)
    mase.append(temp)
    
    
    temp = aqruivo['RMSE'][:len(aqruivo['RMSE'])-3].values
    temp = temp.astype(float)
    rmse.append(temp)
    
    temp = aqruivo['MSE'][:len(aqruivo['MSE'])-3].values
    temp = temp.astype(float)
    mse.append(temp)
    
    temp = aqruivo['MAE'][:len(aqruivo['MAE'])-3].values
    temp = temp.astype(float)
    mae.append(temp)
    
    temp = aqruivo['Tempo'][:len(aqruivo['Tempo'])].values
    temp = temp.astype(float)
    tempos.append(temp)
    
for m in range(len(modelos)):
    rmse_total = []
    smape_total = []
    mase_total = []
    mse_total = []
    mae_total = []
    tempo_total = []
    for s in range(1045):

        smape_total.append(smape[s][m])
        mase_total.append(mase[s][m])
        rmse_total.append(rmse[s][m])
        mse_total.append(mse[s][m])
        mae_total.append(mae[s][m])
        tempo_total.append(tempos[s][m])
        

    
    smape_total = np.array(smape_total)
    mase_total = np.array(mase_total)
    rmse_total = np.array(rmse_total)
    mse_total = np.array(mse_total)
    mae_total = np.array(mae_total)
    tempo_total = np.array(tempo_total)
    line = {'Modelo':modelos[m],
            'SMAPE':round(smape_total.mean(),2),
            'SMAPE (STD)':round(smape_total.std(),2),
            'MASE':round(mase_total.mean(),2),
            'MASE (STD)':round(mase_total.std(),2),
                'RMSE': round(rmse_total.mean(),2),
                'RMSE (STD)': round(rmse_total.std(),2),
                'MSE': round(mse_total.mean(),2),
                'MSE (STD)': round(mse_total.std(),2),
                'MAE': round(mae_total.mean(),2),
                'MAE (STD)': round(mae_total.std(),2),
                'Tempo': round(tempo_total.mean(),2)}
    resultSheetTotal = resultSheetTotal.append(line,ignore_index=True)

tempo_medio_total = []
for s in range(1045):
    tempo_medio_total.append(tempos[s][len(modelos)])
tempo_medio_total = np.array(tempo_medio_total)
line = {'Modelo':'',
                'SMAPE':'',
                'SMAPE (STD)':'',
                'MASE':'',
                'MASE (STD)':'',
                'RMSE': '',
                'RMSE (STD)':'',
                'MSE': '',
                'MSE (STD)': '',
                'MAE': '',
                'MAE (STD)': 'Tempo Médio',
                'Tempo': round(tempo_medio_total.mean(),2) }
    
resultSheetTotal = resultSheetTotal.append(line,ignore_index=True)


tempo_medio_total = []
for s in range(1045):
    tempo_medio_total.append(tempos[s][len(modelos)+2])
tempo_medio_total = np.array(tempo_medio_total)
line = {'Modelo':'',
                'SMAPE':'',
                'SMAPE (STD)':'',
                'MASE':'',
                'MASE (STD)':'',
                'RMSE': '',
                'RMSE (STD)':'',
                'MSE': '',
                'MSE (STD)': '',
                'MAE': '',
                'MAE (STD)': 'Tempo Médio',
                'Tempo': round(tempo_medio_total.mean(),2) }
    
resultSheetTotal = resultSheetTotal.append(line,ignore_index=True)


tempo_medio_total = []
for s in range(1045):
    tempo_medio_total.append(tempos[s][len(modelos)+1])
tempo_medio_total = np.array(tempo_medio_total)
line = {'Modelo':'',
                'SMAPE':'',
                'SMAPE (STD)':'',
                'MASE':'',
                'MASE (STD)':'',
                'RMSE': '',
                'RMSE (STD)':'',
                'MSE': '',
                'MSE (STD)': '',
                'MAE': '',
                'MAE (STD)': 'Tempo Médio',
                'Tempo': round(tempo_medio_total.mean(),2) }
    
resultSheetTotal = resultSheetTotal.append(line,ignore_index=True)

resultSheetTotal.to_excel(excel_writer='Resultado_médio_'+base_dados+'.xlsx',index=False)

