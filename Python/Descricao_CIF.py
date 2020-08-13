# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:38:51 2020

@author: FlavioFilho
"""


import matplotlib.pyplot as plt
import numpy as np
import UtilsCIF 
from util import Utils as ut
import pandas as pd
import matplotlib.pyplot as plt



cif = UtilsCIF.UtilsCIF()



cols = ['Serie','Tamanho','Min','Max','Mean', 'Std','ADI', 'CV']
resultSheet1 = pd.DataFrame(columns=cols)

    

index = cif.listarIndex()
for i in index:
    serie, T = cif.serie(i)
    serie = serie
    
    len(serie)
    print(i)
        
    
    plt.plot(serie)
    plt.title(''+str(i))
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.gcf().autofmt_xdate()
    plt.show()
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
resultSheet1.to_excel(excel_writer='DescricaoBaseCIF_.xlsx',index=False)