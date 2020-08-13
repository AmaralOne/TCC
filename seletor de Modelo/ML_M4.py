# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:55:19 2020

@author: Amaral
"""
import statsmodels.tsa.holtwinters as ts
import pandas as pd
import numpy as np
from numpy.random import seed
seed(42)
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import rmsprop
from keras import backend as ker
from math import sqrt
from util import Utils as ut
import tensorflow as tf
tf.random.set_seed(42)
import gc
import time
from ELM import ELM as elm

import matplotlib.pyplot as plt

class ML_M4:

    def __init__(self, forecating_horizon = 18, frequency = 1, lag = 3):  
        self.fh = forecating_horizon         # forecasting horizon
        self.freq = frequency       # data frequency
        self.in_size = lag    # number of points used as input for each forecast
    
        self.err_MLP_sMAPE = []
        self.err_MLP_MASE = []
        self.err_RNN_sMAPE = []
        self.err_RNN_MASE = []
        self.err_ELM_sMAPE = []
        self.err_ELM_MASE = []
        self.err_Naive = []
        
    
    def acf(self,data, k):
        """
        Autocorrelation function
    
        :param data: time series
        :param k: lag
        :return:
        """
        m = np.mean(data)
        s1 = 0
        for i in range(k, len(data)):
            s1 = s1 + ((data[i] - m) * (data[i - k] - m))
    
        s2 = 0
        for i in range(0, len(data)):
            s2 = s2 + ((data[i] - m) ** 2)
    
        return float(s1 / s2)

    def detrend(self,insample_data):
        """
        Calculates a & b parameters of LRL
    
        :param insample_data:
        :return:
        """
        x = np.arange(len(insample_data))
        a, b = np.polyfit(x, insample_data, 1)
        return a, b
    
    def moving_averages(self,ts_init, window):
        """
        Calculates the moving averages for a given TS
    
        :param ts_init: the original time series
        :param window: window length
        :return: moving averages ts
        """
        """
        As noted by Professor Isidro Lloret Galiana:
        line 82:
        if len(ts_init) % 2 == 0:
        
        should be changed to
        if window % 2 == 0:
        
        This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
        In order for the results to be fully replicable this change is not incorporated into the code below
        """
        
        if window % 2 == 0:
            #ts_ma = pd.rolling_mean(ts_init, window, center=True)
            #ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
            ts = pd.Series(ts_init)
            ts_ma = ts.rolling(window,center=True).mean()
            ts_ma = ts_ma.rolling(2,center=True).mean()
            ts_ma = np.roll(ts_ma, -1)
            
        else:
            #ts_ma = pd.rolling_mean(ts_init, window, center=True)
            ts = pd.Series(ts_init)
            ts_ma = ts.rolling(window,center=True).mean()
    
        return ts_ma
    
    
    def seasonality_test(self,original_ts, ppy):
        """
        Seasonality test
    
        :param original_ts: time series
        :param ppy: periods per year
        :return: boolean value: whether the TS is seasonal
        """
        s = self.acf(original_ts, 1)
        for i in range(2, ppy):
            s = s + (self.acf(original_ts, i) ** 2)
    
        limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))
    
        return (abs(self.acf(original_ts, ppy))) > limit
    
    def deseasonalize(self,original_ts, ppy):
        """
        Calculates and returns seasonal indices
    
        :param original_ts: original data
        :param ppy: periods per year
        :return:
        """
        """
        # === get in-sample data
        original_ts = original_ts[:-out_of_sample]
        """
        if self.seasonality_test(original_ts, ppy):
            # print("seasonal")
            # ==== get moving averages
            ma_ts = self.moving_averages(original_ts, ppy)
    
            # ==== get seasonality indices
            le_ts = original_ts * 100 / ma_ts
            le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
            le_ts = np.reshape(le_ts, (-1, ppy))
            si = np.nanmean(le_ts, 0)
            norm = np.sum(si) / (ppy * 100)
            si = si / norm
        else:
            # print("NOT seasonal")
            si = np.full(ppy, 100)
    
        return si
    
  
    
    def split_into_train_test(self,data, in_num, fh):
        """
        Splits the series into train and test sets. Each step takes multiple points as inputs
    
        :param data: an individual TS
        :param fh: number of out of sample points
        :param in_num: number of input points for the forecast
        :return:
        """
        train, test = data[:-fh], data[-(fh + in_num):]
        x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
        x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]
    
        # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    
        x_train = np.reshape(x_train, (-1, 1))
        x_test = np.reshape(x_test, (-1, 1))
        temp_test = np.roll(x_test, -1)
        temp_train = np.roll(x_train, -1)
        for x in range(1, in_num):
            x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
            x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
            temp_test = np.roll(temp_test, -1)[:-1]
            temp_train = np.roll(temp_train, -1)[:-1]
    
        return x_train, y_train, x_test, y_test
    
    
    def rnn_bench(self,x_train, y_train, x_test, fh, input_size):
        """
        Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer
    
        :param x_train: train data
        :param y_train: target values for training
        :param x_test: test data
        :param fh: forecasting horizon
        :param input_size: number of points used as input
        :return:
        """
        # reshape to match expected input
        x_train = np.reshape(x_train, (-1, input_size, 1))
        x_test = np.reshape(x_test, (-1, input_size, 1))
    
        # create the model
        model = Sequential([
            SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
                      use_bias=False, kernel_initializer='glorot_uniform',
                      recurrent_initializer='orthogonal', bias_initializer='zeros',
                      dropout=0.0, recurrent_dropout=0.0),
            Dense(1, use_bias=True, activation='linear')
        ])
        opt = rmsprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
    
        # fit the model to the training data
        model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)
    
        predct_train = model.predict(x_train)
        
    
        # make predictions
        y_hat_test = []
        last_prediction = model.predict(x_test)[0]
        for i in range(0, fh):
            y_hat_test.append(last_prediction)
            x_test[0] = np.roll(x_test[0], -1)
            x_test[0, (len(x_test[0]) - 1)] = last_prediction
            last_prediction = model.predict(x_test)[0]
            
        ker.clear_session()
    
        return predct_train, np.asarray(y_hat_test)
    
    
    def mlp_bench(self,x_train, y_train, x_test, fh):
        """
        Forecasts using a simple MLP which 6 nodes in the hidden layer
    
        :param x_train: train input data
        :param y_train: target values for training
        :param x_test: test data
        :param fh: forecasting horizon
        :return:
        """
        y_hat_test = []
    
        model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                             max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                             random_state=42)
        
        #model = MLPRegressor(hidden_layer_sizes=9, activation='tanh', solver='adam',
         #                    max_iter=200, learning_rate='adaptive', learning_rate_init=0.001,
          #                   random_state=42)
        model.fit(x_train, y_train)
        

        
        predct_train = model.predict(x_train)
    
        last_prediction = model.predict(x_test)[0]
        for i in range(0, fh):
            y_hat_test.append(last_prediction)
            x_test[0] = np.roll(x_test[0], -1)
            x_test[0, (len(x_test[0]) - 1)] = last_prediction
            last_prediction = model.predict(x_test)[0]
            
        return predct_train, np.asarray(y_hat_test)
    
    def prepararSerie(self,data,horizion=18, frequency = 1, lag = 3):
        
        self.fh = horizion
        self.Data = data
        self.ts = data.values
        self.freq = frequency       # data frequency
        self.in_size = lag    # number of points used as input for each forecast
        
               
        
        #Desanaliza a série temporal
        #self.seasonality_in = self.deseasonalize(self.ts, self.freq)
        self.seasonality_in = self.deseasonalize(self.ts[:-self.fh], self.freq)#maneira corrigida
        
        for i in range(0, len(self.ts)):
            self.ts[i] = self.ts[i] * 100 / self.seasonality_in[i % self.freq]
            
       
        
        #Tira a Tendencia da série temporal
        #self.a, self.b = self.detrend(self.ts)
        self.a, self.b = self.detrend(self.ts[:-self.fh])

        for i in range(0, len(self.ts)):
            self.ts[i] = self.ts[i] - ((self.a * i) + self.b)
            
         
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_into_train_test(self.ts, self.in_size, self.fh)
             
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def desPreprocessarSerie(self, data):
        
        
        #print('des 1')
        #plt.plot(self.ts)
        #plt.plot(data)
        #plt.title('des 1')
        #plt.show()   
        
        # add trend
        #print("Tamanho train: ",len(self.ts))
        #print("Tamanho teste: ",self.fh)
        for i in range(0, len(self.ts)):
            self.ts[i] = self.ts[i] + ((self.a * i) + self.b)

        for i in range(0, self.fh):
            #data[i] = data[i] + ((self.a * (len(self.ts) + i + 1)) + self.b)
            data[i] = data[i] + ((self.a * (len(self.ts) - self.fh + i )) + self.b)
            
        #print('des 2')
        #plt.plot(self.ts)
        #plt.plot(data)
        #plt.title('des 2')
        #plt.show() 
            
        # add seasonality
        for i in range(0, len(self.ts)):
            self.ts[i] = self.ts[i] * self.seasonality_in[i % self.freq] / 100
            
        #daaaa for i in range(len(self.ts), len(self.ts) + self.fh):
            #data[i - len(self.ts)] = data[i - len(self.ts)] * self.seasonality_in[i % self.freq] / 100
            
        for i in range(len(self.ts) - self.fh, len(self.ts) ):
            data[i - (len(self.ts) - self.fh)] = data[i - (len(self.ts) - self.fh)] * self.seasonality_in[i % self.freq] / 100


       
           
        #print('des 3')
        #plt.plot(self.ts)
        #plt.plot(data)
        #plt.title('des 3')
        #plt.show() 

        # check if negative or extreme
        for i in range(len(data)):
            if data[i] < 0:
                data[i] = 0

                
            if data[i] > (100 * max(self.ts)):
                data[i] = max(self.ts)    
                
        #print('des 4')
        #plt.plot(self.ts)
        #plt.plot(data)
        #plt.title('des 4')
        #plt.show() 
        
        return data


    def desPreprocessarSerieTrain(self, data):


        for i in range(0, len(data)):
            data[i] = data[i] + ((self.a * (i + self.in_size)) + self.b)
            

        for i in range(0, len(data)):
            data[i] = data[i] * self.seasonality_in[i % self.freq] / 100

        # check if negative or extreme
        for i in range(len(data)):
            if data[i] < 0:
                data[i] = 0

                
            if data[i] > (1000 * max(self.ts)):
                data[i] = max(self.ts)         
        return data

    def treat_output(self,trainPredict, testPredict):
        
        dataset = self.Data
        index = self.Data.index
        horizion = self.fh
        time_delay = self.in_size
        
        
        train_Index = pd.Series(trainPredict,index[time_delay:len(trainPredict)+time_delay])   
        
        testPredict = np.array(testPredict)
        testPredict = testPredict.reshape((len(dataset)-len(trainPredict)-(time_delay)))
        
        test_Index = pd.Series(testPredict,index[len(trainPredict)+(time_delay):len(dataset)])
           
        
        return train_Index, test_Index
    
    def fit_MLP(self):
        predct_train_MLP, y_hat_test_MLP = self.mlp_bench(self.x_train, self.y_train, self.x_test, self.fh)
        
        for i in range(0, 29):
            predct_train_aux, y_hat_aux = self.mlp_bench(self.x_train, self.y_train, self.x_test, self.fh)
            predct_train_MLP = np.vstack((predct_train_MLP, predct_train_aux))
            y_hat_test_MLP = np.vstack((y_hat_test_MLP, y_hat_aux))
        predct_train_MLP = np.median(predct_train_MLP, axis=0)
        y_hat_test_MLP = np.median(y_hat_test_MLP, axis=0)
        
        
        self.predct_train_MLP = self.desPreprocessarSerieTrain(predct_train_MLP)
        self.y_hat_test_MLP = self.desPreprocessarSerie(y_hat_test_MLP)   
        
        
#        self.err_MLP_sMAPE.append(ut.smape(self.y_test, self.y_hat_test_MLP))
 #       self.err_MLP_MASE.append(ut.mase_ML(self.ts[:-self.fh], self.y_test, self.y_hat_test_MLP, self.freq))
        
        
        self.predct_train_MLP, self.y_hat_test_MLP = self.treat_output(self.predct_train_MLP, self.y_hat_test_MLP)
        
        return self.predct_train_MLP, self.y_hat_test_MLP
    
    def fit_RNN(self):
        
        
        
        predct_train_RNN,y_hat_test_RNN = self.rnn_bench(self.x_train, self.y_train, self.x_test, self.fh, self.in_size) 
        y_hat_test_RNN = np.reshape(y_hat_test_RNN, (-1))
        predct_train_RNN = np.reshape(predct_train_RNN, (-1))
        
        self.predct_train_RNN = self.desPreprocessarSerieTrain(predct_train_RNN)
        self.y_hat_test_RNN = self.desPreprocessarSerie(y_hat_test_RNN)     
        
        
       # self.err_RNN_sMAPE.append(ut.smape(self.y_test, self.y_hat_test_RNN))
       # self.err_RNN_MASE.append(ut.mase_ML(self.ts[:-self.fh], self.y_test, self.y_hat_test_RNN, self.freq))
        
        
        self.predct_train_RNN, self.y_hat_test_RNN = self.treat_output(self.predct_train_RNN, self.y_hat_test_RNN)
        
        return self.predct_train_RNN, self.y_hat_test_RNN
    
    def fit_ELM(self):
        data_aux = pd.Series(self.ts,self.Data.index)
        model = elm(data_aux,3, 30, 0.80, self.fh)
        model.predictions(20)
        #preditc_train = model.trainPredict_Result
        #preditc_test = model.testPredict_Result
        preditc_train = model.trainPredict
        preditc_test = model.testPredict
        self.in_size = model.time_delay()

        
        self.predct_train_ELM = self.desPreprocessarSerieTrain(preditc_train)
        self.y_hat_test_ELM = self.desPreprocessarSerie(preditc_test)     
        
        
        #self.err_ELM_sMAPE.append(ut.smape(self.y_test, self.y_hat_test_ELM))
        #self.err_ELM_MASE.append(ut.mase_ML(self.ts[:-self.fh], self.y_test, self.y_hat_test_ELM, self.freq))
        
        
        self.predct_train_ELM, self.y_hat_test_ELM = self.treat_output(self.predct_train_ELM, self.y_hat_test_ELM)
        
        return self.predct_train_ELM, self.y_hat_test_ELM
    
    def lag(self):
        return True
    
    