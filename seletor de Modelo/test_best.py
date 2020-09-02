# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:21:47 2020

@author: Amaral
"""
import UtilsM3
import time
import numpy as np
import pandas as pd
import Ensembles
import MethodSelector
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")
data12 = ts[:]
tamanho_teste = 18
    
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

method_slector = MethodSelector.MethodSelector()

m = 'ELM'
m = 'MLP A1'

t1 = time.time()
preditc_train, preditc_test, preditc_train_best, preditc_test_best = method_slector.method_Predict(m,ts.copy(),trainData,testData,1)
tempoExec = time.time() - t1

print("Tempo de execução: {} segundos".format(tempoExec))
plt.plot(data12,label='Original')
plt.plot(preditc_test, label=m)
plt.plot(preditc_test_best, label=m+"_best")
plt.legend(loc="upper left")
plt.gcf().autofmt_xdate()
plt.show()