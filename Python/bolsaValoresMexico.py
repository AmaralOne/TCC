# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:00:56 2020

@author: Amaral
"""


import pandas as pd
import matplotlib.pyplot as plt

#file = u'S&P_BMV IPC_Dados Historicos.csv'
#file = u'S&P_BMV_IPC_DadosHistóricos_serie2.csv'
file = u'AUD_INR_Historical_Data.csv'
path = 'dataset/'
freq = 'd'
arquivo = pd.read_csv(path+file,None)

index_arquivo = arquivo.iloc[:,0].values
ts_arquivo = arquivo.iloc[:,1].values

ts_aux = []
index_aux = []
for var in reversed(ts_arquivo):
    ts_aux.append(var.replace('.','').replace(',','.'))
for ind in reversed(index_arquivo):
    index_aux.append(ind)



ts = pd.Series(ts_aux,index_aux,'double')


plt.plot(ts.values)
plt.title('Mexican Stock Exchange')
plt.xlabel('Days')
plt.ylabel('Closing')
plt.savefig('Mexican_Stock_Exchange2.png',dpi = 300)
plt.show()

