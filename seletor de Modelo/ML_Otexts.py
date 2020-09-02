# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:47:45 2020

@author: Amaral
"""


import numpy as np
import pandas as pd
import warnings
from util import Utils as ut
import statsmodels.tsa.ar_model as ar
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import rmsprop
from sklearn.neural_network import MLPRegressor
import statistics as st
from Utils_neural_network import Utils_neural_network as Util_NN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class ML_Otexts:

    def __init__(self):
        warnings.filterwarnings("ignore")
        print()
    
    def get_quantiade_lag(self, trainData):
        
        return ar.AR(trainData).select_order(12,'aic')
        
    def fit(self,data, test_data, horizonte_previsao = 18,subtrain = 0,p = 0,P = 1,size = 0, scale = True, repete = 20, tipo = 'keras'):

        self.p = p
        self.P = P
        self.lags = 0
        self.size = size
        self.scale = scale
        self.frenquencia = self.pegaFrequenciaDaSerie(data)
        self.tamanhoSerie = len(data)
        self.index = data.index        
        self.data = np.array(data)
        self.dataset = data
        self.horizonte_previsao = horizonte_previsao
        self.repete = repete
        self.subtrain = subtrain
        self.tipo = tipo
        self.test_data = test_data

        
        
        
        
        if(len(self.data) <= 3):
            raise NameError("A série Temporal é muito pequena")
            
        if(self.serie_constante(self.data)):
            self.p = 1
            self.P = 0
            self.scale = False
            print("Está série é constante")
            
        if(self.serie_com_valor_vazio(self.data)):
            print("A valores 0 na série temporal")
            
        self.dataset_aux = self.dataset
        if(self.subtrain != 0):
            self.data = self.data[0:self.subtrain]
            self.dataset_aux = self.dataset[0:self.subtrain]
            self.tamanhoSerie = len(self.data)
        
        self.ts = self.data
        if(scale):
            self.ts = self.normaliza_serie(self.ts)
            

        if(self.frenquencia == 1):
            if(self.p == 0):
                #Encontra o melhor lag de acordo com o AR
                self.p = self.get_quantiade_lag(self.ts)
            if(self.p >= self.tamanhoSerie):
                self.p = self.tamanhoSerie - 1
            self.lags = self.p
            if(self.P > 1):
                print('Aviso: A série não é sazonal')
                self.P = 0
        else:
            if(self.p == 0):
                if(self.tamanhoSerie > 2 * self.frenquencia):
                    ts_aux = self.desasonalizar_serie(self.dataset_aux)
                else:
                    ts_aux = self.ts
                self.p = self.get_quantiade_lag(ts_aux)
            if(self.p >= self.tamanhoSerie):
                self.p = self.tamanhoSerie - 1
            if(self.P > 0 and self.tamanhoSerie > self.frenquencia * self.P + 2):
                self.lags = self.p + self.P
            else:
                self.lags = 0
                self.P = 0
        
        if(self.size == 0):
            self.size = round((self.lags + 1)/2)
                   
        #Criar matriz de defasagem
        #Redimensiona os dados para o tamanho do atraso escolhido
        #Divide os dados em X e y
        self.x,self.y = self.criar_atrasos_na_serie_temporal(self.ts,self.p,self.P,self.frenquencia)
        
        if(len(self.y) == 0):
            raise NameError("A série Temporal é muito pequena")
            
        if(self.tipo == 'keras'):
            self.avnnet()
        elif(self.tipo == 'sklearn'):
            self.__fit_MLP()
        elif(self.tipo == 'rnn'):
            self.__fit_RNN()
        
        self.trainPredict = np.array(self.trainPredict)
        self.predict_horizon = np.array(self.predict_horizon)
        
        #the best
        self.trainPredict_best = np.array(self.trainPredict_best)
        self.predict_horizon_best = np.array(self.predict_horizon_best)

        
        self.ts = self.trazer_serie_escala_normal(self.ts)
        self.trainPredict = self.trazer_serie_escala_normal(self.trainPredict)
        self.predict_horizon = self.trazer_serie_escala_normal(self.predict_horizon)
        
        #the best
        self.trainPredict_best = self.trazer_serie_escala_normal(self.trainPredict_best)
        self.predict_horizon_best  = self.trazer_serie_escala_normal(self.predict_horizon_best )

        
        todos_lags = []
        
        for atraso in range(self.p):
            todos_lags.append(atraso+1)
        for atraso_sazonal in range(self.P):
            todos_lags.append((atraso_sazonal+1) * self.frenquencia)
            
        max_lag = max(todos_lags)
        
        self.trainPredict = pd.Series(self.trainPredict,self.index[max_lag:len(self.trainPredict)+max_lag])
        
        #the best
        self.trainPredict_best = pd.Series(self.trainPredict_best,self.index[max_lag:len(self.trainPredict_best)+max_lag])
     
        t= self.index[-1]
        t1 = self.trainPredict.index[-1]
        
        index_h = pd.date_range(start= self.trainPredict.index[-1],
                              freq=self.index.freq,
                              periods = self.horizonte_previsao+1)
        
        self.predict_horizon = np.array(self.predict_horizon)
        self.predict_horizon = self.predict_horizon.reshape((self.horizonte_previsao))
        
        self.predict_horizon = pd.Series(self.predict_horizon,index_h[1:])
        
        #the best
        self.predict_horizon_best = np.array(self.predict_horizon_best)
        self.predict_horizon_best = self.predict_horizon_best.reshape((self.horizonte_previsao))
        
        self.predict_horizon_best = pd.Series(self.predict_horizon_best,index_h[1:])
        
        print("Config: p:",self.p," P:",self.P,"size:",self.size,"F:",self.frenquencia)
             
        return self.trainPredict, self.predict_horizon, self.trainPredict_best, self.predict_horizon_best
    
    
    def criar_atrasos_na_serie_temporal(self,ts,time_delay = 3,atraso_sazonal = 1,freq = 1):
        dataset = np.array(ts)
        dataX, dataY = [], []
        for i in range(len(dataset)-time_delay):
            y_atual = (i+time_delay)
            a = dataset[i:y_atual]
            sazonal = []
            if(freq > 0):            
                for lag_s in range(atraso_sazonal):
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

    def avnnet(self):
        
        list_train = []
        list_predict_horizon = []
        
        indice_best = 0
        current_best = 10000000
        
        median_trainPredict = []
        median_predict_horizon = []
        
        for r in range(self.repete):
            
            # train neural network model  with chosen stopping criterion
            self.__training_model(self.lags,self.size,self.x,self.y)
            

            # generate predictions for training
            self.trainPredict = Util_NN.convert_column_in_row(self.__model.predict(self.x))
            
            horizon_X, horizon_Y = self.__forecating(self.horizonte_previsao,self.lags,self.ts,self.__model)
            
            predict_horizon = horizon_Y

                         
            list_train.append(self.trainPredict)
            list_predict_horizon.append(predict_horizon)
            
            aux_erro = ut.rmse(self.test_data, predict_horizon)
            if(aux_erro < current_best):
                current_best = aux_erro
                indice_best = r
        
        self.trainPredict_best = list_train[indice_best]
        self.predict_horizon_best = list_predict_horizon[indice_best]
                
        #make the median combination of the predictions    
        for x in range(len(self.trainPredict)):
            c = np.array(list_train)
            c = c[:,x]
            c = st.mean(c)
            median_trainPredict.append(c)
            
        for x in range(len(predict_horizon)):
            c = np.array(list_predict_horizon)
            c = c[:,x]
            c = st.mean(c)
            median_predict_horizon.append(c)
                
        self.trainPredict = median_trainPredict
        self.predict_horizon = median_predict_horizon
        
        
    def resultadoModelo(self):
        retorno = "NNAR("+str(self.p)
         
        if(self.P>0):
            retorno = retorno + ","+str(self.P)
             
        retorno = retorno + ","+str(self.size)+")"
        
        if(self.P>0):
            retorno = retorno + "["+str(self.frenquencia)+"]"
        return retorno
        
    def __training_model(self, delay , nodes_hidden, Train__X, Train__Y):
        opt = rmsprop(lr=0.001)
        self.__model = self.__creat_model(delay,nodes_hidden,'adam','sigmoid')                
        #'adam'
        # neural network training will stop when it stops improving after 10 iterations
        #es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
        # will decrease neural network learning rate after stop improving 5 iterations
        #rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
        
        #callbacks = [es,rlr]
        # training model
        self.__model.fit(Train__X, Train__Y, epochs=100, batch_size=1, verbose=0,)
       
    def __forecating(self,horizion, time_delay,dataset, model):
        predictionsX = np.zeros((horizion, time_delay))
        predictionsY = np.zeros((horizion))
        
        ts_aux = dataset
        
        Home_position = len(dataset)-(self.p)
        Final_position = len(dataset)
        predictionsX[0][0:self.p] = dataset[Home_position:Final_position]
        for i in range(self.P):
            predictionsX[0][self.p:(self.p+(i+1))] = dataset[len(dataset) - (self.frenquencia * (i+1))]
        #continuar aqui colocar atraso sazonal
        aux = predictionsX[0].reshape((1, time_delay))
        
  
        for h in range(horizion):
            obs = aux
            if(self.tipo =='rnn'):
                obs = np.reshape(aux, (-1, self.lags, 1))
            predictionsY[h] = model.predict(obs)
            ts_aux = np.concatenate((ts_aux,[predictionsY[h]]), axis=0)
            if h < horizion-1:
                for t in range(self.p-1):
                    predictionsX[h+1][t] = predictionsX[h][t+1]
                for i in range(self.P):
                    predictionsX[h+1][self.p:(self.p+(i+1))] = ts_aux[len(ts_aux) - (self.frenquencia * (i+1))]
        
                predictionsX[h+1][self.p-1] = predictionsY[h]
                aux = predictionsX[h+1].reshape((1, time_delay))
                

        return predictionsX, predictionsY
    
    def mlp_bench(self,x_train, y_train, size_hidden):

        #model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
         #                    max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
          #                   random_state=42)
        
        self.model = MLPRegressor(hidden_layer_sizes=size_hidden, activation='logistic', solver='lbfgs',
                             max_iter=100, learning_rate='adaptive', learning_rate_init=0.001
                             )
        self.model.fit(x_train, y_train)
        #random_state=42
        
        
        predct_train = self.model.predict(x_train)
        
        predictionsX, predictionsY = self.__forecating(self.horizonte_previsao,self.lags,self.ts,self.model)
            
        return predct_train, predictionsY
    
    def rnn_bench(self,x_train, y_train, fh, input_size):



              

        # reshape to match expected input
        x_train = np.reshape(x_train, (-1, input_size, 1))
        
        
    
        # create the model
        self.model = Sequential([
            SimpleRNN(self.size, input_shape=(input_size, 1), activation='linear',
                      use_bias=False, kernel_initializer='glorot_uniform',
                      recurrent_initializer='orthogonal', bias_initializer='zeros',
                      dropout=0.0, recurrent_dropout=0.0),
            Dense(1, use_bias=True, activation='linear')
        ])
        opt = rmsprop(lr=0.001)
        self.model.compile(loss='mean_squared_error', optimizer=opt)
    
        # fit the model to the training data
        self.model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)
    
        predct_train = self.model.predict(x_train)
    
        predictionsX, predictionsY = self.__forecating(self.horizonte_previsao,self.lags,self.ts,self.model)
        

            
        K.clear_session()
        
        return predct_train, predictionsY
            

    def __fit_MLP(self):
        
        indice_best = 0
        current_best = 10000000
        predict_train_best = []
        predict_test_best = []
        
        test_normalizado = self.normaliza_serie_teste(self.test_data.copy())
        predct_train_MLP, y_hat_test_MLP = self.mlp_bench(self.x,self.y, self.size)
        
        #print(f"test_normalizado: {test_normalizado}")
        #print(f"y_hat_test_MLP: {y_hat_test_MLP}")
        aux_erro = ut.rmse(test_normalizado, y_hat_test_MLP)
        if(aux_erro < current_best):
            current_best = aux_erro
            predict_train_best = predct_train_MLP[:]
            predict_test_best = y_hat_test_MLP[:]
        
        for i in range(0, self.repete):
            predct_train_aux, y_hat_aux = self.mlp_bench(self.x,self.y, self.size)
            predct_train_MLP = np.vstack((predct_train_MLP, predct_train_aux))
            y_hat_test_MLP = np.vstack((y_hat_test_MLP, y_hat_aux))
            
            
            aux_erro = ut.rmse(test_normalizado, y_hat_aux)
            if(aux_erro < current_best):
                current_best = aux_erro
                predict_train_best = predct_train_aux
                predict_test_best = y_hat_aux
                
        predct_train_MLP = np.median(predct_train_MLP, axis=0)
        y_hat_test_MLP = np.median(y_hat_test_MLP, axis=0)
        
        self.trainPredict_best = predict_train_best
        self.predict_horizon_best = predict_test_best
        
        self.trainPredict = predct_train_MLP
        self.predict_horizon = y_hat_test_MLP
        
    def __fit_RNN(self):
        

        current_best = 10000000
        predict_train_best = []
        predict_test_best = []
        
        predct_train_RNN,y_hat_test_RNN = self.rnn_bench(self.x,self.y,self.horizonte_previsao,self.lags) 
        
        y_hat_test_RNN = np.reshape(y_hat_test_RNN, (-1))
        predct_train_RNN = np.reshape(predct_train_RNN, (-1))
        test_normalizado = self.normaliza_serie_teste(self.test_data.copy())
        aux_erro = ut.rmse(test_normalizado, y_hat_test_RNN)
        if(aux_erro < current_best):
            current_best = aux_erro
            predict_train_best = predct_train_RNN[:]
            predict_test_best = y_hat_test_RNN[:]
        
        for i in range(0, 9):
            predct_train_RNN_aux,y_hat_test_RNN_aux = self.rnn_bench(self.x,self.y,self.horizonte_previsao,self.lags)
            y_hat_test_RNN_aux = np.reshape(y_hat_test_RNN_aux, (-1))
            predct_train_RNN_aux = np.reshape(predct_train_RNN_aux, (-1))
            predct_train_RNN = np.vstack((predct_train_RNN, predct_train_RNN_aux))
            y_hat_test_RNN = np.vstack((y_hat_test_RNN, y_hat_test_RNN))
            
            aux_erro = ut.rmse(test_normalizado, y_hat_test_RNN_aux)
            if(aux_erro < current_best):
                current_best = aux_erro
                predict_train_best = predct_train_RNN_aux
                predict_test_best = y_hat_test_RNN_aux
                
        predct_train_RNN = np.median(predct_train_RNN, axis=0)
        y_hat_test_RNN = np.median(y_hat_test_RNN, axis=0)
        
        self.trainPredict_best = predict_train_best
        self.predict_horizon_best = predict_test_best
        
        self.trainPredict = predct_train_RNN
        self.predict_horizon = y_hat_test_RNN
           
        

    def __creat_model(self,time_delay,hidden_nodes,optimization_algorithm,activation_function):  
        K.clear_session()
        model = Sequential()
        if activation_function == 'RELU/LINEAR':
            model.add(Dense(hidden_nodes, input_dim=time_delay, activation='relu'))
            model.add(Dense(1,activation='linear'))        
        else:
            model.add(Dense(hidden_nodes, input_dim=time_delay, activation='sigmoid'))
            model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer= optimization_algorithm)
        return model
    
    def serie_constante(self,serie):
        aux = serie[0]
        for s in serie:
            if(s !=aux):
                return False
        return True
    
    def serie_com_valor_vazio(self,serie):
        for s in serie:
            if(s == 0):
                return True
        return False
    
    def normaliza_serie(self, serie):

        serie = serie.reshape(-1,1)
        self.scaler_x = StandardScaler().fit(serie)
        serie = self.scaler_x.transform(serie)
        serie = serie.reshape(-1)
        return serie
    
    def normaliza_serie_teste(self, serie):

        print(len(serie))
        serie = np.array(serie)
        serie = serie.reshape(-1,1)
        serie = self.scaler_x.transform(serie)
        serie = serie.reshape(-1)
        return serie
    
    def trazer_serie_escala_normal(self, serie):

        serie = serie.reshape(-1,1)
        serie = self.scaler_x.inverse_transform(serie)
        serie = serie.reshape(-1)
        return serie
    
    def desasonalizar_serie(self, serie):
        
        # Time Series Decomposition
        result_mul = seasonal_decompose(serie, model='additive', extrapolate_trend=self.frenquencia)
        # Deseasonalize
        deseasonalized = serie - result_mul.seasonal
        
        return deseasonalized
    
    def tratarOutliners(self,serie):
        # check if negative or extreme
        for i in range(len(serie)):
            if serie[i] < 0:
                serie[i] = 0

        return serie
    
    def transformar_dados_para_elm(self):
        # reshape dataset for elm
        dataTrain = np.zeros(((len(self.x)),(self.lags+1)))
        dataTrain[:,1:(self.lags+1)] = self.x
        dataTrain[:,0] = self.y
        
        return dataTrain
        
    
    def pegaFrequenciaDaSerie(self,serie):
        result = serie.index.freq

        
        if(result == 'A' or result == 'Y'):
            return 1
        elif(result == 'Q'):
            return 4
        elif(result == 'D'):
            return 7
        elif(result == 'M'):
            return 12
        elif(result == 'W'):
            return 52
 
        
        return 1