# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:23:04 2020

@author: Amaral
"""


import UtilsCIF
import pandas as pd
import matplotlib.pyplot as plt
import Ensembles

cif = UtilsCIF.UtilsCIF()
index = cif.listarIndex()
index = index[64:]

modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'RNN A1','RNN A2','RNN A3',
                'ELM']

intervalo_dos_estocasticos = 11


modelos_finais = modelos[:]


modelos_finais.append('NNAR_best')
modelos_finais.append('NNAR RNN_best')
modelos_finais.append('MLP A1_best')
modelos_finais.append('MLP A2_best')
modelos_finais.append('MLP A3_best')
modelos_finais.append('RNN A1_best')
modelos_finais.append('RNN A2_best')
modelos_finais.append('RNN A3_best')
modelos_finais.append('ELM_best')

modelos_finais.append('Comb Media')
modelos_finais.append('Comb Mediana')
modelos_finais.append('Comb Media Aparada')
modelos_finais.append('Comb Media Ponderada')

modelos_finais.append('Comb Media best')
modelos_finais.append('Comb Mediana best')
modelos_finais.append('Comb Media Aparada best')
modelos_finais.append('Comb Media Ponderada best')

for serie in index:
    #serie = 'ts54'
    print(serie)
    ts, tamanho_teste = cif.serie(serie)
    #Imprimir Gráfico da Séreie
    plt.plot(ts)
    plt.title('Série '+serie)
    plt.show()
    
    
    
    #Dividir a Série Temporal em treino e Teste
    tamanho_serie = len(ts)
    inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
    incio_de_teste = (tamanho_serie-tamanho_teste)
    trainData = ts[:inico_de_validacao]
    validationData = ts[inico_de_validacao:]
    testData = ts[incio_de_teste:]


    e = Ensembles.Ensembles()
    results, train_models, tempoExecModelos = e.Ensembles_predict('',modelos,ts,trainData,validationData,intervalo_dos_estocasticos,tamanho_teste)
    
    cols_train = ["Modelo"]
    for p in range(len(trainData)):
        cols_train.append(str(p))
    train_models_reuslt = pd.DataFrame(columns=cols_train)
    for m in modelos_finais:
        if 'Comb' in m:
            continue
        line_train = {}
        line_train['Modelo'] = m
        inicio = 0
        if len(train_models[m]) < len(trainData):
            inicio = len(trainData) - len(train_models[m])
        for p in range(len(trainData)):
            if p+inicio == len(trainData):
               break
            line_train[str(p+inicio)] = train_models[m][p]
        train_models_reuslt = train_models_reuslt.append(line_train,ignore_index=True)
            

    
    if(tamanho_teste == 11):
        cols = ['Modelo','1','2','3','4','5','6','7','8','9','10','11']
        
        predict_models_validation = pd.DataFrame(columns=cols)
        predict_models_test = pd.DataFrame(columns=cols)
        for m in modelos_finais:
            line_validation = {'Modelo':m,
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

                            
                        }
            line_test = {'Modelo':m,
                            '1': results[m][11],
                            '2': results[m][12],
                            '3': results[m][13],
                            '4': results[m][14],
                            '5': results[m][15],
                            '6': results[m][16],
                            '7': results[m][17],
                            '8': results[m][18],
                            '9': results[m][19],
                            '10': results[m][20],
                            '11': results[m][21],

                            
                        }
            predict_models_validation = predict_models_validation.append(line_validation,ignore_index=True)
            predict_models_test = predict_models_test.append(line_test,ignore_index=True)
    else:
        cols = ['Modelo','1','2','3','4','5','6']
        predict_models_validation = pd.DataFrame(columns=cols)
        predict_models_test = pd.DataFrame(columns=cols)
        for m in modelos_finais:
            line_validation = {'Modelo':m,
                            '1': results[m][0],
                            '2': results[m][1],
                            '3': results[m][2],
                            '4': results[m][3],
                            '5': results[m][4],
                            '6': results[m][5],
                        }
            line_test = {'Modelo':m,
                            '1': results[m][6],
                            '2': results[m][7],
                            '3': results[m][8],
                            '4': results[m][9],
                            '5': results[m][10],
                            '6': results[m][11],
                        }
            predict_models_validation = predict_models_validation.append(line_validation,ignore_index=True)
            predict_models_test = predict_models_test.append(line_test,ignore_index=True)
    
        
    writer = pd.ExcelWriter('Resut_cif\Resultado_Predict_'+serie+'.xlsx', engine='xlsxwriter')
    predict_models_validation.to_excel(writer,sheet_name='predict_validation',index=False)
    predict_models_test.to_excel(writer,sheet_name='predict_test',index=False)
    train_models_reuslt.to_excel(writer,sheet_name='train',index=False)
    model_time = pd.Series(tempoExecModelos, index=tempoExecModelos.keys())
    model_time.to_excel(writer,sheet_name='tempo')
    
    writer.save()