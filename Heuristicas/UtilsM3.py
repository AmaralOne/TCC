# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:22:27 2020

@author: FlavioFilho
"""

import pandas as pd



class UtilsM3:
    #Cronstrutor da classe
    def __init__(self):
        self.file = u'M3C.xls'
        self.path = 'dataset/'
        self.freq = 'M'
        self.sheet_names = ['2017','2018','2019']
        self.prop=80
        self.aqruivo = pd.read_excel(self.path+self.file,None)
        self.aqruivo = self.aqruivo.pop('M3Month')
        #self.aqruivo = self.aqruivo[self.aqruivo['N']>=126]
        
    def listarIndex(self):

        names = self.aqruivo['Series'].unique()
        names.sort()
        return names
        
    def buildM3DataFrame(self,seriesName,size=None):
        '''
            Get monthly data from M3 data set and return a Pandas.Series.
        '''
        
        # Get specific row
        newDF = self.aqruivo[self.aqruivo['Series']==seriesName]
       
        # To initialize date
        
        startYear = str(newDF['Starting Year'].values[0])
        month = str(newDF['Starting Month'].values[0])
        if startYear == '0':
            startYear = '1995'
            month = '1'
        
        startDate = month+'/'+startYear
        if size==None:
            size = newDF['N'].values[0]
            
        # Create new series
        dates = pd.date_range(start=startDate,periods=size,freq='M') #index
        values = newDF.iloc[0,6:size+6].values #values
        ts = pd.Series(values,dates,'double')
        
        return ts

 

            