# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:23:04 2020

@author: Amaral
"""


import UtilsCIF
import UtilsBIO
import pandas as pd
import matplotlib.pyplot as plt
import Ensemble_Retreino

cif = UtilsCIF.UtilsCIF()
index = cif.listarIndex()
bio = UtilsBIO.UtilsBio()
index = bio.listarProdutos()
index = index[:]

modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
               'RNN A1','RNN A2','RNN A3',
                'ELM']

intervalo_dos_estocasticos = 11


modelos_finais = modelos[:]



modelos_finais.append('Comb Media')
modelos_finais.append('Comb Mediana')
modelos_finais.append('Comb Media Aparada')
modelos_finais.append('Comb Media Ponderada')



for serie in index:
    #serie = 'ts54'
    print(serie)
    ts = bio.read_Dataset_BIO(serie)
    tamanho_teste = (int)((len(ts))*0.2)
    
    tamanho_serie = len(ts)
    
    if tamanho_serie < 30:
        continue
    
    zero = False
    for i in range(len(ts)):
        if ts[i] == 0:
            zero = True
            break
    
    if zero == True:
        continue
    
    #Imprimir Gráfico da Séreie
    plt.plot(ts)
    plt.title('Série '+str(serie))
    plt.show()
    
    tamanho_validacao = (int)((len(ts))*0.2)
    
    #Dividir a Série Temporal em treino e Teste
    tamanho_serie = len(ts)
    #inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
    inico_de_validacao = (tamanho_serie-(tamanho_teste+tamanho_validacao))
    incio_de_teste = (tamanho_serie-tamanho_teste)
    trainData = ts[:incio_de_teste]
    validationData = ts[inico_de_validacao:]
    testData = ts[incio_de_teste:]
    
    #result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_'+serie+'.xlsx',None)
    result_series = pd.read_excel('D:\\Projetos\\TCC\\seletor de Modelo\\Result_bio\\Resultado_Predict_20%'+str(serie)+'.xlsx',None)
    result_validation = result_series.pop('predict_validation')

    e = Ensemble_Retreino.Ensemble_Retreino()
    results, train_models, tempoExecModelos = e.Ensembles_predict('',modelos,ts,trainData,testData,intervalo_dos_estocasticos,tamanho_validacao,result_validation)
    
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
        
        
    cols_teste = ["Modelo"]
    for p in range(tamanho_teste):
        cols_teste.append(str(p))
    predict_models_test = pd.DataFrame(columns=cols_teste)
    for m in modelos_finais:
        line_test = {}
        line_test['Modelo'] = m
        for p in range((tamanho_teste)):
            line_test[str(p)] = results[m][p]
        predict_models_test = predict_models_test.append(line_test,ignore_index=True)
            

    
    #if(tamanho_teste == 11):
        #cols = ['Modelo','1','2','3','4','5','6','7','8','9','10','11']
        

        #predict_models_test = pd.DataFrame(columns=cols)
        #for m in modelos_finais:
            #line_test = {'Modelo':m,
                           # '1': results[m][0],
                           # '2': results[m][1],
                           # '3': results[m][2],
                           # '4': results[m][3],
                           # '5': results[m][4],
                           # '6': results[m][5],
                           # '7': results[m][6],
                           # '8': results[m][7],
                            #'9': results[m][8],
                           # '10': results[m][9],
                          #  '11': results[m][10],

                            
                 #       }
            
            #predict_models_test = predict_models_test.append(line_test,ignore_index=True)
    #else:
        #cols = ['Modelo','1','2','3','4','5','6']
        #predict_models_test = pd.DataFrame(columns=cols)
        #for m in modelos_finais:
            #line_test = {'Modelo':m,
                         #   '1': results[m][0],
                        #    '2': results[m][1],
                       #     '3': results[m][2],
                      #      '4': results[m][3],
                     #       '5': results[m][4],
                    #        '6': results[m][5],
                   #     }
           
            #predict_models_test = predict_models_test.append(line_test,ignore_index=True)
    
        
    writer = pd.ExcelWriter('Result_bio_Retreino\Resultado_Predict_retreino20%_'+str(serie)+'.xlsx', engine='xlsxwriter')
    predict_models_test.to_excel(writer,sheet_name='predict_test',index=False)
    train_models_reuslt.to_excel(writer,sheet_name='train',index=False)
    model_time = pd.Series(tempoExecModelos, index=tempoExecModelos.keys())
    model_time.to_excel(writer,sheet_name='tempo')
    
    writer.save()