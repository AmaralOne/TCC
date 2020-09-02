# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:12:56 2020

@author: Amaral
"""
import Predicts_Erro
import UtilsCIF
import pandas as pd


predict_erros = Predicts_Erro.Predicts_Erro()
cif = UtilsCIF.UtilsCIF()
index = cif.listarIndex()


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
reuslt_validation_smape = pd.DataFrame(columns=cols)
reuslt_test_smape = pd.DataFrame(columns=cols)
    

for serie in index:
    
    ts, tamanho_teste = cif.serie(serie)
    
    #Dividir a Série Temporal em treino e Teste
    tamanho_serie = len(ts)
    inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
    incio_de_teste = (tamanho_serie-tamanho_teste)
    trainData = ts[:incio_de_teste]
    validationData = ts[inico_de_validacao:incio_de_teste]
    testData = ts[incio_de_teste:]
    
    #result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_'+serie+'.xlsx',None)
    result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif_Retreino\\Resultado_Predict_retreino_'+serie+'.xlsx',None)
    
    result_validation = result_series.pop('predict_validation')
    result_prediction = result_series.pop('predict_test')
    
    results_v = {}
    for m in range(len(result_validation)):
        if result_validation.iloc[m][0] not in modelos:
            continue           
        results_v[result_validation.iloc[m][0]] = pd.Series(result_validation.iloc[m][1:])
    
    results_t = {}
    for m in range(len(result_prediction)):
        if result_prediction.iloc[m][0] not in modelos:
            continue           
        results_t[result_prediction.iloc[m][0]] = pd.Series(result_prediction.iloc[m][1:])
    
    
    forecast_errors_validation_rmse = predict_erros.error_RMSE(modelos,results_v,validationData)
    forecast_errors_test_rmse = predict_erros.error_RMSE(modelos,results_t,testData)
    forecast_errors_validation_smape = predict_erros.error_SMAPE(modelos,results_v,validationData)
    forecast_errors_test_smape = predict_erros.error_SMAPE(modelos,results_t,testData)
    

    line_validation_rmse = {}
    line_test_rmse = {}
    line_validation_smape = {}
    line_test_smape = {}
    
    line_validation_rmse['Série'] = serie
    line_test_rmse['Série'] = serie
    line_validation_smape['Série'] = serie
    line_test_smape['Série'] = serie
    
    for m in modelos:
        line_validation_rmse[m] = round(forecast_errors_validation_rmse[m],3)
        line_test_rmse[m] = round(forecast_errors_test_rmse[m],3)
        line_validation_smape[m] = round(forecast_errors_validation_smape[m],3)
        line_test_smape[m] = round(forecast_errors_test_smape[m],3)
        
    reuslt_validation_rmse = reuslt_validation_rmse.append(line_validation_rmse,ignore_index=True)
    reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse,ignore_index=True)
    reuslt_validation_smape = reuslt_validation_smape.append(line_validation_smape,ignore_index=True)
    reuslt_test_smape = reuslt_test_smape.append(line_test_smape,ignore_index=True)

line_validation_rmse_mean = {}
line_test_rmse_mean = {}
line_validation_smape_mean = {}
line_test_smape_mean = {}

line_validation_rmse_std = {}
line_test_rmse_std = {}
line_validation_smape_std = {}
line_test_smape_std = {}          

line_validation_rmse_mean['Série'] = "Media"
line_test_rmse_mean['Série'] = "Media"
line_validation_smape_mean['Série'] = "Media"
line_test_smape_mean['Série'] = "Media"

line_validation_rmse_std['Série'] = "std"
line_test_rmse_std['Série'] = "std"
line_validation_smape_std['Série'] = "std"
line_test_smape_std['Série'] = "std"


for m in modelos:
    line_validation_rmse_mean[m] = round(reuslt_validation_rmse[m].mean(),3)
    line_test_rmse_mean[m] = round(reuslt_test_rmse[m].mean(),3)
    line_validation_smape_mean[m] = round(reuslt_validation_smape[m].mean(),3)
    line_test_smape_mean[m] = round(reuslt_test_smape[m].mean(),3)
    
    line_validation_rmse_std[m] = round(reuslt_validation_rmse[m].std(),3)
    line_test_rmse_std[m] = round(reuslt_test_rmse[m].std(),3)
    line_validation_smape_std[m] = round(reuslt_validation_smape[m].std(),3)
    line_test_smape_std[m] = round(reuslt_test_smape[m].std(),3)

        
reuslt_validation_rmse = reuslt_validation_rmse.append(line_validation_rmse_mean,ignore_index=True)
reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse_mean,ignore_index=True)
reuslt_validation_smape = reuslt_validation_smape.append(line_validation_smape_mean,ignore_index=True)
reuslt_test_smape = reuslt_test_smape.append(line_test_smape_mean,ignore_index=True)

reuslt_validation_rmse = reuslt_validation_rmse.append(line_validation_rmse_std,ignore_index=True)
reuslt_test_rmse = reuslt_test_rmse.append(line_test_rmse_std,ignore_index=True)
reuslt_validation_smape = reuslt_validation_smape.append(line_validation_smape_std,ignore_index=True)
reuslt_test_smape = reuslt_test_smape.append(line_test_smape_std,ignore_index=True)



#writer = pd.ExcelWriter('CIF_ERROS.xlsx', engine='xlsxwriter')
writer = pd.ExcelWriter('CIF_ERROS_retreino.xlsx', engine='xlsxwriter')
reuslt_validation_rmse.to_excel(writer,sheet_name='validation_rmse',index=False)
reuslt_test_rmse.to_excel(writer,sheet_name='test_rmse',index=False)
reuslt_validation_smape.to_excel(writer,sheet_name='validation_smape',index=False)
reuslt_test_smape.to_excel(writer,sheet_name='test_smape',index=False)
writer.save()