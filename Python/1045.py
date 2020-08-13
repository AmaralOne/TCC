#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

dataFile = 'M3C.xls'
path = 'dataset/'
data = pd.read_excel(path+dataFile,None)
datatemp = pd.DataFrame()
sheets = ['M3Month']#['M3Year','M3Quart','M3Month','M3Other']

for sheet in sheets:
    
    dataAux = data.pop(sheet)
    sheetAux =  dataAux.loc[(dataAux['N']-18>80)]
    datatemp = datatemp.append(sheetAux)
   
print(len(datatemp))