# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:27:06 2020

@author: Amaral
"""
import pandas as pd
import numpy as np


  

# MA example
from statsmodels.tsa.arima_model import ARMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARMA(data, order=(0, 20))
model_fit = model.fit(disp=False)

#yhat = model_fit.predict(0, len(data)+15)

testPredictions = pd.Series(np.ceil(model_fit.predict(
                start=len(data),
                end=len(data)+18-1,
                dynamic=False)))
print(testPredictions)

