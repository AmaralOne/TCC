# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:32:26 2019

@author: FlavioFilho
"""

import pandas as pd
from datetime import datetime
class UtilsRFA:
    
    dataSetsPath = None
    dataFrameTable = None
    serie = None
    
    #Cronstrutor da classe
    def __init__(self,dataSetsPath,fileName):
        self.dataSetsPath = dataSetsPath
        self.__fileToDataFrame(fileName)
        
        
    
    def __str__(self):
        return str(self.dataFrameTable)

    #Ler o arquivo
    def __fileToDataFrame(self,fileName):
        
        fileType = fileName[fileName.rfind('.')+1:]
        
        if fileType == 'xlsx':
            self.dataFrameTable = pd.read_excel(self.dataSetsPath+fileName)
        elif fileType == 'csv':
            self.dataFrameTable = pd.read_csv(self.dataSetsPath+fileName) 
        else:
            raise ("Unexpected file type!")
    #Retorna a serie temporal escolhida
    def getTimeSeries(self,itemName):
        
        #Verifica se o item existente esta no intervalo das series existentes
        if(itemName > 5000 or itemName <= 0):
            raise ("There is no such time series")
        
        #Seleciona a serie tempral escolhida
        Item = self.dataFrameTable[self.dataFrameTable['Item Ref no'] == itemName].index
        Item = self.dataFrameTable.iloc[Item] 
        self.serie = Item.iloc[:,4:] 
        serieValues = self.serie.values
        #Trabalhar as datas da serie Temporal
        AnoInicial = 1996
        AnoFinal = 2002
        Date = self.__getIntervaloData(AnoInicial, AnoFinal)
        NewSerie = pd.Series(serieValues[0])
        NewSerie.index = Date
        
        return NewSerie
    
    #Retorna um array com dos meses do ano Inicial até o ano Final
    def __getIntervaloData(self,AnoInicial, AnoFinal):
        Datas = []
        
        while(AnoInicial <= AnoFinal):
            Mes = 1
            while(Mes <= 12):
                if(Mes >= 10):
                    data_e_hora_em_texto = '01/'+str(Mes)+'/'+str(AnoInicial)+' 00:00'
                    Datas.append(datetime.strptime(data_e_hora_em_texto, '%d/%m/%Y %H:%M'))
                else:
                    data_e_hora_em_texto = '01/0'+str(Mes)+'/'+str(AnoInicial)+' 00:00'
                    Datas.append(datetime.strptime(data_e_hora_em_texto, '%d/%m/%Y %H:%M'))
                Mes = Mes + 1
            AnoInicial = AnoInicial + 1
        return Datas
                    
                    
                



