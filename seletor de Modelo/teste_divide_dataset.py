# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:16:32 2020

@author: Amaral
"""

import UtilsM3
import numpy as np

#Carregar uma Série Temporal do Conjunto M3
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")



def criar_atrasos_na_serie_temporal(ts,time_delay = 3,atraso_sazonal = 1,freq = 1):
    dataset = np.array(ts)
    dataX, dataY = [], []
    for i in range(len(dataset)-time_delay):
        y_atual = (i+time_delay)
        print('Y atual: ',y_atual)
        a = dataset[i:y_atual]
        sazonal = []
        if(freq > 0):            
            for lag_s in range(atraso_sazonal):
                print('Resultado: ',y_atual-(freq*(lag_s+1)))
                if(y_atual-(freq*(lag_s+1)) >= 0):
                    sazonal.append(dataset[y_atual-(freq*(lag_s+1))])
                else:
                    sazonal.append(0.0)
                         
        s = np.array(sazonal)
        a = np.concatenate((a,s), axis=0)
        if(0 not in s):
            dataX.append(a)
            dataY.append(dataset[y_atual])
        
    x = np.array(dataX)
    y = np.array(dataY)
    
    return x,y

from sklearn.preprocessing import StandardScaler
ts = np.array(ts)
ts = ts.reshape(-1,1)
scaler_x = StandardScaler().fit(ts)
ts = scaler_x.transform(ts)
ts = ts.reshape(-1)
x,y = criar_atrasos_na_serie_temporal(ts,freq=0)



def divide_dados_de_treion_e_teste(dataX,dataY,percentage=0.80, horizonte = 18):

    if horizonte == 0:
        horizonte = len(dataset) - int(len(dataset) * percentage)
            
    train_x = x[:len(x)-horizonte,:]
    train_y = y[:len(y)-horizonte]
    teste_x = x[len(x)-horizonte:,:]
    teste_y = y[len(y)-horizonte:]
    return train_x, train_y, teste_x, teste_y

train_x, train_y, teste_x, teste_y = divide_dados_de_treion_e_teste(x,y,0.8,18)