# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:54:10 2020

@author: Amaral
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#file = u'S&P_BMV IPC_Dados Historicos.csv'
file = u'cif.csv'
file_result = u'cif_result.csv'
path = 'dataset/'
freq = 'd'
arquivo = pd.read_csv(path+file,None)
arquivo_result = pd.read_csv(path+file_result,None)


ts = arquivo.iloc[0,3:].values
ts_h = arquivo_result.iloc[0,1:].values

ts = np.reshape(ts, (-1, 1))
ts_h = np.reshape(ts_h, (-1, 1))


ts_final = np.concatenate([ts,ts_h])


plt.plot(ts)
plt.plot(ts_final)
plt.show()