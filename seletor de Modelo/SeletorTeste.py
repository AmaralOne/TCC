# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:41:46 2020

@author: Amaral
"""
import pandas as pd
import matplotlib.pyplot as plt
from util import Utils as ut
import time
import UtilsM3
import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import Utils as ut
import Ses
import Naive
import Holt
import AR
import Croston2 as cr
import ML_M4
import autoSVR
import ARIMA
import ML_Otexts
from ELM import ELM as elm
import operator
import sklearn
import warnings
import autoMA
warnings.filterwarnings("ignore")

#serie = "N1679"
#Carregar uma Série Temporal do Conjunto M3




#file = u'AUD_INR_Historical_Data.csv'
#file = u'FB_Historical_Data.csv'
#path = 'dataset/'
#freq = 'd'
#arquivo = pd.read_csv(path+file,None)

#index_arquivo = arquivo.iloc[:,0].values
#ts_arquivo = arquivo.iloc[:,1].values

#ts_aux = []
#index_aux = []
#for var in reversed(ts_arquivo):
#    ts_aux.append(var)
#for ind in reversed(index_arquivo):
#    index_aux.append(ind)


#ts = pd.Series(ts_aux,index_aux,'double')

import UtilsM3_100

import UtilsCIF

base_dados = 'sub_M3'

cols = ['Modelo','SMAPE', 'SMAPE (STD)','MASE', 'MASE (STD)','RMSE', 'RMSE (STD)','MSE','MSE (STD)','MAE','MAE (STD)','Tempo']
resultSheetTotal = pd.DataFrame(columns=cols)

smape = []
mase = []
rmse = []
mse = []
mae = []
tempos = []

#U_m3 = UtilsM3_100.UtilsM3_100()
#index = U_m3.listarIndex()
#index = index[0:]
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
#cif = UtilsCIF.UtilsCIF()
#index = cif.listarIndex()
index = index[800]

for serie in index:
    print(serie)
    ts = U_m3.buildM3DataFrame('N1679')
    
    if(not(len(ts)>= 99) ):
        continue
    
    print('Série '+serie)
    #ts, tamanho_teste = cif.serie(serie)
    #Imprimir Gráfico da Séreie
    plt.plot(ts)
    plt.title('Série '+serie)
    plt.show()
    
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
    
    RESULTADO = []
    result = {}
    tempoExecModelos = {}
    
    ts_aux_2 = ts
    
    
    for m in modelos:
        
        if(m == 'ses'):
            print(m)
            t1 = time.time()
            model = Ses.Ses()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
        elif (m == 'naive'):
            print(m)
            t1 = time.time()
            model = Naive.Naive()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'holt'):
            print(m)
            t1 = time.time()
            model = Naive.Naive()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'Ar'):
            print(m)
            t1 = time.time()
            model = AR.AR()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'Croston'):
            print(m)
            t1 = time.time()
            model = cr.Croston2()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'Ma'):
            print(m)
            t1 = time.time()
            model = autoMA.autoMA()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'Arima'):
            print(m)
            t1 = time.time()
            model = ARIMA.ARIMA()
            preditc_train = model.fit(trainData)
            preditc_test = model.forecasts(testData)
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
        elif (m == 'SVR A1'):
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,1)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        
            
        elif (m == 'SVR A2'):
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,2)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
    
        
        elif (m == 'SVR A3'):
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,3)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
    
        
        elif (m == 'SVR A4'):
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,4)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
            
        elif (m == 'SVR A5'):
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,5)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
            
        elif (m == 'SVR A6'): 
            print(m)
            t1 = time.time()
            model = autoSVR.autoSVR()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,6)
            preditc_train, preditc_test = model.fit()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
            
        elif (m == 'NNAR'):
            print(m)
            t1 = time.time()
            model = ML_Otexts.ML_Otexts()
            preditc_train, preditc_test = model.fit(ts_aux_2,tamanho_teste,incio_de_teste,0,1,0,True,20,'sklearn')
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
            
            
        elif (m == 'NNAR RNN'): 
            print(m)
            t1 = time.time()
            model = ML_Otexts.ML_Otexts()
            preditc_train, preditc_test = model.fit(ts_aux_2,tamanho_teste,incio_de_teste,0,1,0,True,20,'rnn')
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
             
            
        elif (m == 'MLP A1'):
            print(m)
            t1 = time.time()   
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,1)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'MLP A2'):
            print(m)
            t1 = time.time()     
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,2)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'MLP A3'):
            print(m)
            t1 = time.time()      
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,3)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'MLP A4'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,4)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'MLP A5'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,5)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'MLP A6'):
            print(m)
            t1 = time.time()      
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,6)
            preditc_train, preditc_test = model.fit_MLP()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A1'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,1)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A2'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,2)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A3'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,3)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A4'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,4)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A5'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,5)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'RNN A6'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,6)
            preditc_train, preditc_test = model.fit_RNN()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
        elif (m == 'ELM'):
            print(m)
            t1 = time.time()
            model = ML_M4.ML_M4()
            model.prepararSerie(ts_aux_2,tamanho_teste,1,1)
            preditc_train, preditc_test = model.fit_ELM()
            tempoExec = time.time() - t1
            tempoExecModelos[m] = tempoExec
            print("Tempo de execução: {} segundos".format(tempoExec))
            RESULTADO.append(preditc_test)
            result[m] = preditc_test
    
    
    
    comb_media = []
    comb_mediana = []
        
    for x in range(tamanho_teste):
        total = []    
        for m in range(len(modelos)):
            aux = RESULTADO[m]
            #acrecentou um iloc para a serie do facebook
            total.append(aux.iloc[x])
        total = np.array(total)   
        c_mediana = st.median(total)  
        c_media = total.mean()
        comb_mediana.append(c_mediana)
        comb_media.append(c_media)
            
    result_comb_mediana = pd.Series(comb_mediana,ts.index[len(ts)-tamanho_teste:])
    result_comb_media = pd.Series(comb_media,ts.index[len(ts)-tamanho_teste:])
    
    dictList = []
    for key, value in tempoExecModelos.items():
        dictList.append(value)
    tempoModelos = np.array(dictList)
    tempo_medio = tempoModelos.mean()
    tempo_total = tempoModelos.sum()
    devio_padrão_tempo = tempoModelos.std()
    
    
    #if(model.lag()):
    #    initialValues = ts[:len(trainData)-len(preditc_train)]
     #   preditc_train = initialValues.append(preditc_train)
    
    freq = 12
    
    cols = ['Modelo','SMAPE','MASE','RMSE','MSE','MAE','Tempo']
    resultSheet1 = pd.DataFrame(columns=cols)
    erros = []
    erros_aux = {}
    erros_mse = {}
    erros_mae = {}
    erros_smape = {}
    erros_mase = {}
    for m in range(len(modelos)):
        
        Erro_RMSE_Test = ut.rmse(testData,RESULTADO[m])
        Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,RESULTADO[m])
        Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,RESULTADO[m])
        Erro_SMPAE_Test = ut.smape(testData, RESULTADO[m])

        Erro_MASE_Test = ut.mase_ML(trainData, testData, RESULTADO[m], freq)
        erros.append(Erro_MSE_Test)
        erros_aux[modelos[m]] = Erro_RMSE_Test
        erros_mse[modelos[m]] = Erro_MSE_Test
        erros_mae[modelos[m]] = Erro_MAE_Test
        erros_smape[modelos[m]] = Erro_SMPAE_Test
        erros_mase[modelos[m]] = Erro_MASE_Test
        
        
        line = {'Modelo':modelos[m],
                'SMAPE': round(Erro_SMPAE_Test,2),
                'MASE': round(Erro_MASE_Test,2),
                'RMSE': round(Erro_RMSE_Test,2),
                'MSE': round(Erro_MSE_Test,2),
                'MAE': round(Erro_MAE_Test,2),
                'Tempo': round(tempoExecModelos[modelos[m]],2)}
        resultSheet1 = resultSheet1.append(line,ignore_index=True)
        print(line)
    
    
    
    
    
    
    erros = np.array(erros)
    e = erros **-1
    total_erro = e.sum();
    aux_results = []
    comb_media_ponderada = []
    total = 0
    for m in range(len(modelos)):
        w = ((erros[m]**-1)/total_erro)
        total = total + w
        w1 =w **-1
        plt.plot(RESULTADO[m])
        aux = w * RESULTADO[m]
        aux_results.append(aux)
    for x in range(tamanho_teste):
        total = []    
        for m in range(len(modelos)):
            aux = aux_results[m]
            total.append(aux.iloc[x])
        total = np.array(total)   
        comb_media_ponderada.append(total.sum())
    resutl_comb_media_ponderada = pd.Series(comb_media_ponderada,ts.index[len(ts)-tamanho_teste:])
    
    Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,result_comb_mediana)
    Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,result_comb_mediana)
    Erro_RMSE_Test = ut.rmse(testData,result_comb_mediana)   
    Erro_SMPAE_Test = ut.smape(testData,result_comb_mediana)
    Erro_MASE_Test = ut.mase_ML(trainData, testData, result_comb_mediana, freq)
    line = {'Modelo':'Comb Mediana',
                'SMAPE': round(Erro_SMPAE_Test,2),
                'MASE': round(Erro_MASE_Test,2),
                'RMSE': round(Erro_RMSE_Test,2),
                'MSE': round(Erro_MSE_Test,2),
                'MAE': round(Erro_MAE_Test,2),
                'Tempo': round(0.0,2)}
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    print(line)
    
    Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,result_comb_media)
    Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,result_comb_media)
    Erro_RMSE_Test = ut.rmse(testData,result_comb_media)   
    Erro_SMPAE_Test = ut.smape(testData,result_comb_media)
    Erro_MASE_Test = ut.mase_ML(trainData, testData, result_comb_media, freq)
    line = {'Modelo':'Comb Média',
                'SMAPE': round(Erro_SMPAE_Test,2),
                'MASE': round(Erro_MASE_Test,2),
                'RMSE': round(Erro_RMSE_Test,2),
                'MSE': round(Erro_MSE_Test,2),
                'MAE': round(Erro_MAE_Test,2),
                'Tempo': round(0,2)}
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    print(line)
    
    Erro_MSE_Test = sklearn.metrics.mean_squared_error(testData,resutl_comb_media_ponderada)
    Erro_MAE_Test = sklearn.metrics.mean_absolute_error(testData,resutl_comb_media_ponderada)
    Erro_RMSE_Test = ut.rmse(testData,resutl_comb_media_ponderada)   
    Erro_SMPAE_Test = ut.smape(testData,resutl_comb_media_ponderada)
    Erro_MASE_Test = ut.mase_ML(trainData, testData, resutl_comb_media_ponderada, freq)
    line = {'Modelo':'Comb Média Ponderada',
            'SMAPE': round(Erro_SMPAE_Test,2),
            'MASE': round(Erro_MASE_Test,2),
                'RMSE': round(Erro_RMSE_Test,2),
                'MSE': round(Erro_MSE_Test,2),
                'MAE': round(Erro_MAE_Test,2),
                'Tempo': round(0.0,2)}
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    print(line)
    
    line = {'Modelo':'',
                'SMAPE':'' ,
                'MASE': '' ,
                'RMSE':'' ,
                'MSE': '',
                'MAE': 'Tempo Médio',
                'Tempo': round(tempo_medio,2) }
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    
    line = {'Modelo':'',
               'SMAPE':'' ,
               'MASE': '' ,
                'RMSE':'' ,
                'MSE': '',
                'MAE': 'Tempo Total',
                'Tempo': round(tempo_total,2) }
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    
    line = {'Modelo':'',
                'SMAPE':'' ,
                'MASE': '' ,
                'RMSE':'' ,
                'MSE': '',
                'MAE': 'Tempo STD',
                'Tempo': round(devio_padrão_tempo,2) }
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    
    
    
    
    
    
    sortedDict = sorted(erros_aux.items(), key=operator.itemgetter(1))
    
    
    ts_aux_2 = ts_aux_2.iloc[len(ts_aux_2)-25:]
    result_comb_mediana = result_comb_mediana.iloc[len(ts_aux_2)-25:]
    result_comb_media = result_comb_media.iloc[len(ts_aux_2)-25:]
    resutl_comb_media_ponderada = resutl_comb_media_ponderada.iloc[len(ts_aux_2)-25:]
    #Imprimir Resultados
    plt.plot(ts_aux_2,label='Original')
    for m in range(len(modelos)-17):
        a = sortedDict[m]
        plt.plot(result[a[0]], label=a[0])
    plt.plot(result_comb_mediana, label='Comb Mediana')    
    #plt.plot(result_comb_media, label='Comb Média')   
    plt.plot(resutl_comb_media_ponderada, label='Comb Média Ponderada')  
    plt.legend(loc="upper left")
    plt.gcf().autofmt_xdate()
    plt.title('Série '+serie)
    plt.show()
    
    resultSheet1.to_excel(excel_writer='Resultado_'+base_dados+'_'+serie+'.xlsx',index=False)
    
    temp = resultSheet1['SMAPE'][:len(resultSheet1['SMAPE'])-3].values
    temp = temp.astype(float)
    smape.append(temp)
    
    temp = resultSheet1['MASE'][:len(resultSheet1['MASE'])-3].values
    temp = temp.astype(float)
    mase.append(temp)
    
    
    temp = resultSheet1['RMSE'][:len(resultSheet1['RMSE'])-3].values
    temp = temp.astype(float)
    rmse.append(temp)
    
    temp = resultSheet1['MSE'][:len(resultSheet1['MSE'])-3].values
    temp = temp.astype(float)
    mse.append(temp)
    
    temp = resultSheet1['MAE'][:len(resultSheet1['MAE'])-3].values
    temp = temp.astype(float)
    mae.append(temp)
    
    temp = resultSheet1['Tempo'][:len(resultSheet1['Tempo'])].values
    temp = temp.astype(float)
    tempos.append(temp)

modelos.append('Comb Mediana')
modelos.append('Comb Média')
modelos.append('Comb Média Ponderada')
for m in range(len(modelos)):
    rmse_total = []
    smape_total = []
    mase_total = []
    mse_total = []
    mae_total = []
    tempo_total = []
    for s in range(len(index)):
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
for s in range(len(index)):
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
for s in range(len(index)):
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
for s in range(len(index)):
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

