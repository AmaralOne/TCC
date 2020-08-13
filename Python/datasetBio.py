# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:46:14 2020

@author: FlavioFilho
"""
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import datetime
import math
from util import Utils as ut


# Data definition, global scope
file = u'bio.xlsx'
path = 'dataset/'
freq = 'M'
aqruivo = pd.read_excel(path+file)

def listarProdutos():
    d = datetime.date(2016,6,30)
    produtos = aqruivo[aqruivo['Data'] <= d ]
    produtos = produtos['Codigo']
    produtos = produtos.unique()
    return produtos
    


def read_Dataset_BIO(codigo_produto):   
    newArquivo = aqruivo[aqruivo['Codigo'] == codigo_produto]
    timeSeries = pd.Series(data=newArquivo['Qtd'].values,
                              index=newArquivo['Data'].values)
    timeSeries =  timeSeries.resample(freq)
    timeSeries = timeSeries.sum()
    return timeSeries

codigo_produto = 21 
serie = read_Dataset_BIO(144)
len(serie)

plt.plot(serie)
plt.title('Codigo Produto '+str(codigo_produto))
plt.xlabel('Time')
plt.ylabel('Demand')
plt.gcf().autofmt_xdate()
plt.show()

produtos = listarProdutos()

cols = ['Produto','Tamanho','Min','Max','Mean', 'Std','ADI', 'CV']
resultSheet1 = pd.DataFrame(columns=cols)
        

for p in produtos:
    serie = read_Dataset_BIO(p)

    plt.plot(serie)
    plt.title('Codigo Produto '+str(p))
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.gcf().autofmt_xdate()
    plt.show()
    result = ut.getTimeSeriesStats(serie)
    
    line = {'Produto':p,
                        'Tamanho':len(serie),
                        'Min':result['min'][0],
                        'Max':result['max'][0],
                        'Mean':result['mean'][0],
                        'Std':result['std'][0],
                        'ADI':result['ADI'][0],
                        'CV':result['CV'][0]}
    resultSheet1 = resultSheet1.append(line,ignore_index=True)
    print(result)
    
line = {'Produto':'Média',
                        'Tamanho':resultSheet1['Tamanho'].mean(),
                        'Min':resultSheet1['Min'].mean(),
                        'Max':resultSheet1['Max'].mean(),
                        'Mean':resultSheet1['Mean'].mean(),
                        'Std':resultSheet1['Std'].mean(),
                        'ADI':resultSheet1['ADI'].mean(),
                        'CV':resultSheet1['CV'].mean()}

resultSheet1 = resultSheet1.append(line,ignore_index=True)

line = {'Produto':'STD',
                        'Tamanho':resultSheet1['Tamanho'].std(),
                        'Min':resultSheet1['Min'].std(),
                        'Max':resultSheet1['Max'].std(),
                        'Mean':resultSheet1['Mean'].std(),
                        'Std':resultSheet1['Std'].std(),
                        'ADI':resultSheet1['ADI'].std(),
                        'CV':resultSheet1['CV'].std()}

resultSheet1 = resultSheet1.append(line,ignore_index=True)
resultSheet1.to_excel(excel_writer='DescricaoBaseBio.xlsx',index=False)
    
    
#Verificar Produtos
#334
#242
#3
#2
#1    
    

