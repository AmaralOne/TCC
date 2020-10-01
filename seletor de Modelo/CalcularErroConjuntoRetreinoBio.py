# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:12:56 2020

@author: Amaral
"""
import Predicts_Erro
import UtilsBIO
import pandas as pd
import math


predict_erros = Predicts_Erro.Predicts_Erro()
bio = UtilsBIO.UtilsBio()
index = bio.listarProdutos()


modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'RNN A1','RNN A2','RNN A3',
                'ELM']

modelos.append('NNAR_best')
modelos.append('NNAR RNN_best')
modelos.append('MLP A1_best')
modelos.append('MLP A2_best')
modelos.append('MLP A3_best')
modelos.append('RNN A1_best')
modelos.append('RNN A2_best')
modelos.append('RNN A3_best')
modelos.append('ELM_best')
modelos.append('Comb Media')
modelos.append('Comb Mediana')
modelos.append('Comb Media Aparada')
modelos.append('Comb Media Ponderada')
modelos.append('Comb Media best')
modelos.append('Comb Mediana best')
modelos.append('Comb Media Aparada best')
modelos.append('Comb Media Ponderada best')

cols = ['Série']
cols.extend(modelos)

reuslt_validation_rmse = pd.DataFrame(columns=cols)
reuslt_test_rmse = pd.DataFrame(columns=cols)
reuslt_validation_mase = pd.DataFrame(columns=cols)
reuslt_test_mase = pd.DataFrame(columns=cols)
    

for serie in index:
    
    ts = bio.read_Dataset_BIO(serie)
    tamanho_teste = 6
    
    tamanho_serie = len(ts)
    
    if tamanho_serie < 30:
        continue
    
    print(serie)
    #Dividir a Série Temporal em treino e Teste
    tamanho_serie = len(ts)
    inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
    incio_de_teste = (tamanho_serie-tamanho_teste)
    trainData = ts[:incio_de_teste]
    validationData = ts[inico_de_validacao:incio_de_teste]
    testData = ts[incio_de_teste:]
    
    #result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_'+serie+'.xlsx',None)
    #result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif_Retreino\\Resultado_Predict_retreino_novo_'+serie+'.xlsx',None)
    
    result_series = pd.read_excel('Result_bio_Retreino\Resultado_Predict_retreino2_'+str(serie)+'.xlsx',None)
    

    result_prediction = result_series.pop('predict_test')
    

    results_t = {}
    for m in range(len(result_prediction)):
        if result_prediction.iloc[m][0] not in modelos:
            continue           
        
        aux = pd.Series(result_prediction.iloc[m][1:])
        
        for x in range(len(aux)):
            if math.isnan(aux[x]):
                aux[x] = 0
        print(aux)
        results_t[result_prediction.iloc[m][0]] = aux
        
        
    
    

    forecast_errors_test_rmse = predict_erros.error_RMSE(modelos,results_t,testData.values)
    forecast_errors_test_mase = predict_erros.error_MASE(modelos,results_t,testData.values,trainData.values)
    


    line_test_rmse = {}
    line_test_mase = {}
    

    line_test_rmse['Série'] = serie
    line_test_mase['Série'] = serie
    
    for m in modelos:

        line_test_rmse[m] = round(forecast_errors_test_rmse[m],3)
        line_test_mase[m] = round(forecast_errors_test_mase[m],3)
        

    reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse,ignore_index=True)
    reuslt_test_mase = reuslt_test_mase.append(line_test_mase,ignore_index=True)


line_test_rmse_mean = {}
line_test_mase_mean = {}


line_test_rmse_std = {}
line_test_mase_std = {}          


line_test_rmse_mean['Série'] = "Media"
line_test_mase_mean['Série'] = "Media"


line_test_rmse_std['Série'] = "std"
line_test_mase_std['Série'] = "std"


for m in modelos:

    line_test_rmse_mean[m] = round(reuslt_test_rmse[m].mean(),3)
    line_test_mase_mean[m] = round(reuslt_test_mase[m].mean(),3)
    

    line_test_rmse_std[m] = round(reuslt_test_rmse[m].std(),3)
    line_test_mase_std[m] = round(reuslt_test_mase[m].std(),3)

        

reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse_mean,ignore_index=True)
reuslt_test_mase = reuslt_test_mase.append(line_test_mase_mean,ignore_index=True)


reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse_std,ignore_index=True)
reuslt_test_mase = reuslt_test_mase.append(line_test_mase_std,ignore_index=True)



#writer = pd.ExcelWriter('CIF_ERROS.xlsx', engine='xlsxwriter')
writer = pd.ExcelWriter('CIF_ERROS_retreino_novo2.xlsx', engine='xlsxwriter')
reuslt_test_rmse.to_excel(writer,sheet_name='test_rmse',index=False)
reuslt_test_mase.to_excel(writer,sheet_name='test_mase',index=False)
writer.save()