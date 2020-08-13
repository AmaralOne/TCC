# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:31:54 2020

@author: Amaral
"""

import pandas as pd
import matplotlib.pyplot as plt
import UtilsCIF
from ELM import ELM as elm
import ML_M4

cif = UtilsCIF.UtilsCIF()

ts, tamanho_teste = cif.serie('ts55')
#Imprimir Gráfico da Séreie
plt.plot(ts)
plt.title('ts55')
plt.show()

model = elm(ts,3, 30, 0.80, tamanho_teste)
model.predictions(3)
preditc_train = model.trainPredict
preditc_test = model.testPredict

print('teste')

model = ML_M4.ML_M4()
model.prepararSerie(ts,tamanho_teste,1,1)
preditc_train, preditc_test = model.fit_ELM()