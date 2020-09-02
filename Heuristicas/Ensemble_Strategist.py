# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:00:21 2020

@author: Amaral
"""
import numpy as np
import pandas as pd
import statistics as st

class Ensemble_Strategist:

    def __init__(self):     
        print("")
        
    def Mean_Combination(self,ts,test_size,methods,results):
        comb_media = []  
        
        for x in range(test_size):
            total = []    
            for m in methods:
                aux = results[m]
                total.append(aux.iloc[x])
            total = np.array(total)    
            c_media = total.mean()
            
            comb_media.append(c_media)
                
        result_comb_media = pd.Series(comb_media,ts.index[len(ts)-test_size:])
        return result_comb_media
    
    def Median_Combination(self,ts,test_size,methods,results):
        comb_mediana = [] 
        
        for x in range(test_size):
            total = []    
            for m in methods:
                aux = results[m]
                total.append(aux.iloc[x])
            total = np.array(total)   
            c_mediana = st.median(total)  
            comb_mediana.append(c_mediana)
                
        result_comb_mediana = pd.Series(comb_mediana,ts.index[len(ts)-test_size:])
        return result_comb_mediana
    
    def Trimmed_Mean_Combination(self,ts,test_size,methods,results):
        comb_media_aparada = [] 
        
        for x in range(test_size):
            total = []    
            for m in methods:
                aux = results[m]
                total.append(aux.iloc[x])
            total = np.array(total)
            if len(methods) != 1:
                c_media_aparada = self.media_aparada(total,20)  
            else:
                c_media_aparada = total
            comb_media_aparada.append(c_media_aparada)
                
        resut_comb_media_aparada = pd.Series(comb_media_aparada,ts.index[len(ts)-test_size:])
        return resut_comb_media_aparada
    
    def weighted_average_Combination(self,ts,test_size,methods,results,errors):
        erros = {}
        for m in methods:
            erros[m] = errors[m]
        erros = list(erros.values())
        erros = np.array(erros)
        e = erros **-1
        total_erro = e.sum();
        aux_results = []
        comb_media_ponderada = []
        total = 0
        for m in range(len(methods)):
            w = ((erros[m]**-1)/total_erro)
            total = total + w
            aux = w * results[methods[m]]
            
            aux_results.append(aux)
        for x in range(test_size):
            total = []    
            for m in range(len(methods)):
                aux = aux_results[m]
                total.append(aux.iloc[x])
            total = np.array(total)   
            comb_media_ponderada.append(total.sum())
        resutl_comb_media_ponderada = pd.Series(comb_media_ponderada,ts.index[len(ts)-test_size:])
        return resutl_comb_media_ponderada
    
    
    def media_aparada(self,arr, percent):
        n = len(arr)
        arr = sorted(arr)
        k = int(round(n*(float(percent)/100)/2))
        return np.mean(arr[k+1:n-k])