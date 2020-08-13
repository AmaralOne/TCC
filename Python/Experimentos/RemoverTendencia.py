# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:10:01 2020

@author: FlavioFilho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detrend(insample_data):
    """
    Calculates a & b parameters of LRL

    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b





x = pd.Series([1,3,4,0,9,11,14,17,18,17,20])

plt.plot(x)

    


#Encontrar o ajsuste do modelo da regressão linear
a, b = detrend(x)
print("Coeficiente: ",a)
print("b: ",b)


#Encontar as previsões da regressao linear para as observações
predit = []
for i in range(len(x)):
    predit.append( ((a * i) + b))
    
y = pd.Series(predit)
plt.plot(y)
plt.plot(x)

# Remover o valor das predições nas observações para tirar a tendencia da série
newX = x - y
plt.plot(newX)


# Adicionar o valor das predições nas observação para retornar o valar das tendecia
velhoX = newX + y
plt.plot(velhoX)