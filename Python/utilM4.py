# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:52:56 2020

@author: FlavioFilho
"""

import pandas as pd
import numpy as np

class UtilsM4:
    
    Path_train = None
    Path_test = None
    file_train = None
    file_test =None
    arquivo_train = None
    arquivo_test = None
    serie = None
    
    #Cronstrutor da classe
    def __init__(self,Path_train, Path_test,file_train, file_test):
        self.Path_train = Path_train
        self.Path_test = Path_test
        self.file_train = file_train
        self.file_test = file_test
        
        self.arquivo_train = pd.read_csv(self.Path_train+self.file_train,keep_default_na=False)
        self.arquivo_Teste = pd.read_csv(self.Path_test+self.file_test,keep_default_na=False)

        
    def getIndex(self):
        index = self.arquivo_train['V1']
        return index
    
    def getSerie(self,itemName):
        Item = self.arquivo_train[self.arquivo_train['V1'] == itemName].index
        Item = self.arquivo_train.iloc[Item] 
        I = Item.iloc[0,1:].values
        I = I[I != '']
        I = I.astype(float)
        serie = pd.Series(I) 
        return serie
    
    def getSerieTeste(self,itemName):
        Item_test = self.arquivo_Teste[self.arquivo_Teste['V1'] == itemName].index
        Item_test = self.arquivo_Teste.iloc[Item_test] 
        I_t = Item_test.iloc[0,1:].values
        I_t = I_t[I_t != '']
        I_t = I_t.astype(float)
        serie = pd.Series(I_t) 
        return serie

    def getSerieCompleto(self,itemName):
        Item = self.arquivo_train[self.arquivo_train['V1'] == itemName].index
        Item = self.arquivo_train.iloc[Item] 
        
        Item_test = self.arquivo_Teste[self.arquivo_Teste['V1'] == itemName].index
        Item_test = self.arquivo_Teste.iloc[Item_test] 
        
        I = Item.iloc[0,1:].values
        I = I[I != '']
        I = I.astype(float)
        
        
        I_t = Item_test.iloc[0,1:].values
        I_t = I_t[I_t != '']
        I_t = I_t.astype(float)
        
        I_completo = np.concatenate((I, I_t))
        
        serie = pd.Series(I_completo) 
        return serie
