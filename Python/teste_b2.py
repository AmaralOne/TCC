# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:41:07 2020

@author: FlavioFilho
"""
import utilM4 as u
from math import sqrt
import numpy as np
import pandas as pd

day = {'Nome':'Day','Path_train': 'dataset/Dataset_m4/Train/','Path_test': 'dataset/Dataset_m4/Test/','file_train': u'Daily-train.csv','file_test': u'Daily-test.csv'}
u_M4 = u.UtilsM4(day['Path_train'],day['Path_test'],day['file_train'],day['file_test'])

data = u_M4.getSerieCompleto('D1')

data = data.values

#data_all = np.array(np.random.random_integers(0, 100, (100, 20)), dtype=np.float32)
#for i in range(0, 100):
 #   for j in range(0, 20):
  #      data_all[i, j] = j * 10 + data_all[i, j]
        
#data = data_all[1, :]

len(data)

fh = 18         # forecasting horizon
freq = 1       # data frequency
in_num = 3    # number of points used as input for each forecast

train, test = data[:-fh], data[-(fh + in_num):]
x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

len(test)

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
print(x_train)
print()
print(len(x_train))
x_train = np.reshape(x_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1))
temp_test = np.roll(x_test, -1)
temp_train = np.roll(x_train, -1)
for x in range(1, in_num):
    x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
    x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
    temp_test = np.roll(temp_test, -1)[:-1]
    temp_train = np.roll(temp_train, -1)[:-1]