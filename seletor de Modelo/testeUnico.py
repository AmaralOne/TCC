# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:35:55 2020

@author: Amaral
"""


import UtilsM3
import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import Utils as ut

import operator
import sklearn


#Carregar uma Série Temporal do Conjunto M3
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")

sr = pd.to_datetime(ts) 
result = sr.dt.freq 
print(result)


plt.plot(ts)
plt.title('1')
plt.show()

import ML_M4
        
model = ML_M4.ML_M4()
model.prepararSerie(ts,18,1,5)
preditc_train, preditc_test = model.fit_MLP()


plt.plot(ts)
plt.plot(preditc_train)
plt.plot(preditc_test)
plt.title('Resultado')
plt.show()

ts = U_m3.buildM3DataFrame("N1641")
tamanho_teste = 18

#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

import autoAM

model = autoAM.autoAM()
preditc_train = model.fit(trainData)
preditc_test = model.forecasts(testData)

plt.plot(ts)
plt.plot(preditc_train)
plt.plot(preditc_test)
plt.title('AM')
plt.show()

import AR

model = AR.AR()
preditc_train = model.fit(trainData)
preditc_test = model.forecasts(testData)

plt.plot(ts)
plt.plot(preditc_train)
plt.plot(preditc_test)
plt.title('Ar')
plt.show()


from statsmodels.tsa.ar_model import ar_select_order
mod = ar_select_order(trainData, maxlag=13)
print(mod.ar_lags)
from statsmodels.tsa.ar_model import AutoReg
res = AutoReg(trainData, lags = mod.ar_lags,seasonal = True).fit()
predic = res.predict(len(trainData),len(trainData)+18-1, False,)
predic_train = res.fittedvalues

plt.plot(ts)
plt.plot(predic_train)
plt.plot(predic)
plt.title('Ar novo')
plt.show()


import ARIMA
len(trainData)
model = ARIMA.ARIMA()
preditc_train_arima = model.fit(trainData)
preditc_test_arima = model.forecasts(testData)

plt.plot(ts)
plt.plot(preditc_train_arima)
plt.plot(preditc_test_arima)
plt.title('Arima')
plt.show()

import statsmodels.tsa.ar_model as ar

ar.AR.select_order(maxlag = 13, ic = 'aic', trend='c', method='mle')

def __get_quantiade_lag(trainData):
        
        return ar.AR(trainData).select_order(12,'aic')
__get_quantiade_lag(trainData)


from statsmodels.tsa.seasonal import STL
res = STL(ts,seasonal = 15).fit()
res.plot()
plt.show()

s = res.seasonal
r = res.resid
t = res.trend
ts_menos_tendencia = ts - t
ts_menos_sazonalidade = ts - s
ts_menos_tendencia_sazonalidade = ts - s -t
r_vezes_t = t * r
plt.plot(ts,label='original')
plt.plot(ts_menos_sazonalidade,label='menos sazonalidade')

#plt.plot(ts_menos_tendencia,label='menos tendencia')
#plt.plot(ts_menos_tendencia_sazonalidade, label='menos sazonalidade e tendencia')
#plt.plot(r,label='Resido')
#plt.plot(r_vezes_t,label='Resido x Tendencia')
plt.legend()
plt.title('Resultado da Decomposição')
plt.show()




