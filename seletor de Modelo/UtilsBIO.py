# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:54:13 2020

@author: FlavioFilho
"""

import pandas as pd

import datetime

class UtilsBio:
    #Cronstrutor da classe
    def __init__(self):
        self.file = u'bio.xlsx'
        self.path = 'dataset/'
        self.freq = 'M'
        self.aqruivo = pd.read_excel(self.path+self.file)
        
    def listarProdutos(self):
        d = datetime.date(2016,6,30)
        produtos = self.aqruivo[self.aqruivo['Data'] <= d ]
        produtos = produtos['Codigo']
        produtos = produtos.unique()
        return produtos
    


    def read_Dataset_BIO(self,codigo_produto):   
        newArquivo = self.aqruivo[self.aqruivo['Codigo'] == codigo_produto]
        timeSeries = pd.Series(data=newArquivo['Qtd'].values,
                                  index=newArquivo['Data'].values)
        timeSeries =  timeSeries.resample(self.freq)
        timeSeries = timeSeries.sum()
        return timeSeries