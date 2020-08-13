# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:48:48 2019

@author: FlavioFilho
"""
import UtilsRFA as u
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from util import Utils as ut

dataSetPath = 'dataset/'
fileName = 'RAFdata.xlsx'

Teste = u.UtilsRFA(dataSetPath,fileName)
ts = Teste.getTimeSeries(8)
plt.plot(ts)

cols = ['Serie','Tamanho','Min','Max','Mean', 'Std','ADI', 'CV']
resultSheet1 = pd.DataFrame(columns=cols)

for i in range(1,5001):
    print(i)
    ts = Teste.getTimeSeries(i)
    #plt.plot(ts)
    result = ut.getTimeSeriesStats(ts)
    
    line = {'Serie':str(i),
                        'Tamanho':len(ts),
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
resultSheet1.to_excel(excel_writer='DescricaoBaseRAF.xlsx',index=False)