# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:48:17 2020

@author: Amaral
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import Utils as ut

#file = u'S&P_BMV IPC_Dados Historicos.csv'
file = u'river.csv'
path = 'dataset/'
freq = 'd'
arquivo = pd.read_csv(path+file,None)

ts = arquivo.iloc[0:,0].values
ts = pd.Series(ts)
result = ut.getTimeSeriesStats(ts)
    
line = {'Serie':'i',
                            'Tamanho':len(ts),
                            'Min':result['min'][0],
                            'Max':result['max'][0],
                            'Mean':result['mean'][0],
                            'Std':result['std'][0],
                            'ADI':result['ADI'][0],
                            'CV':result['CV'][0]}
print(line)

plt.plot(ts)
plt.title('River flow')
plt.xlabel('Months')
plt.ylabel('River flow')
plt.savefig('Mexican_Stock_Exchange2.png',dpi = 300)
plt.show()