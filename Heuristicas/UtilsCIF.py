# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:22:27 2020

@author: FlavioFilho
"""

import pandas as pd
import numpy as np



class UtilsCIF:
    #Cronstrutor da classe
    def __init__(self):
        self.file = u'cif.csv'
        self.file_result = u'cif_result.csv'
        self.path = 'dataset/'
        self.freq = 'M'
        
        self.arquivo = pd.read_csv(self.path+self.file,None)
        self.arquivo_result = pd.read_csv(self.path+self.file_result,None)
        
        
    def listarIndex(self):

        names = self.arquivo['serie'].unique()
        names.sort()
        return names
        
    def serie_treino(self,seriesName,size=None):

        
        # Get specific row
        newDF = self.arquivo[self.arquivo['serie']==seriesName]

        # Create new series
        values = newDF.iloc[0,3:].values
        dates = pd.date_range(start=1,periods=len(values),freq='M') #index
        ts = pd.Series(values,dates,'double')
        
        return ts
    
    def serie_teste(self,seriesName,size=None):

        
        # Get specific row
        newDF = self.arquivo_result[self.arquivo_result['serie']==seriesName]

        # Create new series
        values = newDF.iloc[0,1:].values
        dates = pd.date_range(start=1,periods=len(values),freq='M') #index
        ts = pd.Series(values,dates,'double')
        
        return ts
    
    def serie(self,seriesName,size=None):
        

        
        # Get specific row
        newDF_train = self.arquivo[self.arquivo['serie']==seriesName]
        newDF_result = self.arquivo_result[self.arquivo_result['serie']==seriesName]

        # Create new series
        values_train = newDF_train.iloc[0,3:].values
        values_teste = newDF_result.iloc[0,1:].values
        
        values_train = self.tiraNulos(values_train)
        values_teste = self.tiraNulos(values_teste)
        
        
        
        h = len(values_teste)
        
        values_train = np.reshape(values_train, (-1, 1))
        values_teste = np.reshape(values_teste, (-1, 1))


        values = np.concatenate([values_train,values_teste])
        values = np.reshape(values, (-1))


        dates = pd.date_range(start=1,periods=len(values),freq='M') #index
        ts = pd.Series(values,dates,'double')
        
        return ts, h
    
    def tiraNulos(self,serie):
        for i in range(len(serie)):
            if(pd.isna(serie[i])):
                break
        return serie[0:i]

 

            