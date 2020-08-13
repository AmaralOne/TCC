# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:37:20 2020

@author: Amaral
"""


import UtilsM3
import matplotlib.pyplot as plt


#Carregar uma Série Temporal do Conjunto M3
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")
tamanho_teste = 18

#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

import ML_Otexts
model = ML_Otexts.ML_Otexts()
trainPredict, predict_horizon = model.fit(ts,18,incio_de_teste,0,1,0,True,20,'rnn')

plt.plot(ts)
plt.plot(trainPredict)
plt.plot(predict_horizon)
plt.title(model.resultadoModelo())
plt.show()