# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:38:51 2020

@author: FlavioFilho
"""


import matplotlib.pyplot as plt
import numpy as np
import utilM4 as u
from util import Utils as ut
import pandas as pd



file_M4 = []
    
day = {'Nome':'Day','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Daily-train.csv','file_test': u'Daily-test.csv'}
horas = {'Nome':'Horas','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Hourly-train.csv','file_test': u'Hourly-test.csv'}
mes = {'Nome':'Mes','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Monthly-train.csv','file_test': u'Monthly-test.csv'}
quarto = {'Nome':'Quarto','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Quarterly-train.csv','file_test': u'Quarterly-test.csv'}
semana = {'Nome':'Semana','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Weekly-train.csv','file_test': u'Weekly-test.csv'}
ano = {'Nome':'Ano','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Yearly-train.csv','file_test': u'Yearly-test.csv'}
 
file_M4.append(day)
file_M4.append(horas)
file_M4.append(mes)
file_M4.append(quarto)
file_M4.append(semana)
file_M4.append(ano)   


for arquivo in file_M4:
    u_M4 = u.UtilsM4(arquivo['Path_train'],arquivo['Path_test'],arquivo['file_train'],arquivo['file_test'])
    
    cols = ['Serie','Tamanho','Min','Max','Mean', 'Std','ADI', 'CV']
    resultSheet1 = pd.DataFrame(columns=cols)

    
    index = u_M4.getIndex()
    for i in index:
        serie = u_M4.getSerieCompleto(i)
        len(serie)
        print(i)
        
    
        #plt.plot(serie)
        #plt.title(''+str(i))
        #plt.xlabel('Time')
        #plt.ylabel('Demand')
        #plt.gcf().autofmt_xdate()
        #plt.show()
        result = ut.getTimeSeriesStats(serie)
    
        line = {'Serie':str(i),
                            'Tamanho':len(serie),
                            'Min':result['min'][0],
                            'Max':result['max'][0],
                            'Mean':result['mean'][0],
                            'Std':result['std'][0],
                            'ADI':result['ADI'][0],
                            'CV':result['CV'][0]}
        resultSheet1 = resultSheet1.append(line,ignore_index=True)
        print(result)
     
    line = {'Serie':'Média',
                            'Tamanho':resultSheet1['Tamanho'].mean(),
                            'Min':resultSheet1['Min'].mean(),
                            'Max':resultSheet1['Max'].mean(),
                            'Mean':resultSheet1['Mean'].mean(),
                            'Std':resultSheet1['Std'].mean(),
                            'ADI':resultSheet1['ADI'].mean(),
                            'CV':resultSheet1['CV'].mean()}
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    
    line = {'Serie':'STD',
                            'Tamanho':resultSheet1['Tamanho'].std(),
                            'Min':resultSheet1['Min'].std(),
                            'Max':resultSheet1['Max'].std(),
                            'Mean':resultSheet1['Mean'].std(),
                            'Std':resultSheet1['Std'].std(),
                            'ADI':resultSheet1['ADI'].std(),
                            'CV':resultSheet1['CV'].std()}
    
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    resultSheet1.to_excel(excel_writer='DescricaoBaseM4_'+arquivo['Nome']+'.xlsx',index=False)