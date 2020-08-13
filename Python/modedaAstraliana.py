# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:04:16 2020

@author: Amaral
"""


# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:00:56 2020

@author: Amaral
"""


import pandas as pd
import matplotlib.pyplot as plt


#file = u'AUD_INR_Historical_Data.csv'
file = u'FB_Historical_Data.csv'
path = 'dataset/'
freq = 'd'
arquivo = pd.read_csv(path+file,None)

index_arquivo = arquivo.iloc[:,0].values
ts_arquivo = arquivo.iloc[:,1].values

ts_aux = []
index_aux = []
for var in reversed(ts_arquivo):
    ts_aux.append(var)
for ind in reversed(index_arquivo):
    index_aux.append(ind)


ts = pd.Series(ts_aux,index_aux,'double')
from util import Utils as ut
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

plt.plot(ts.values)
plt.title('AUD-INR exchange rate')
plt.xlabel('Months')
plt.ylabel('CExchange rate')
plt.savefig('Mexican_Stock_Exchange2.png',dpi = 300)
plt.show()