# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:17:20 2020

@author: Amaral
"""


import MethodSelector
import Ensemble_Strategist
import Predicts_Erro
import UtilsM3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ensembles:

    def __init__(self):  
       
        print("")
        self.result = {}
        self.result_train = {}
        self.tempoExecModelos = {}
        
        self.method_slector = MethodSelector.MethodSelector()
        self.ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
        self.predict_erros = Predicts_Erro.Predicts_Erro()
             
        
    def Ensembles_predict(self,ensembles, methods, data,tain_data,test_data,intervalo_dos_estocasticos, tamanhoValidacao):
        i = 0
        plt.plot(data,label='Original')
        for m in methods:
            print(m)
            ts = data.copy()
            t1 = time.time()
            print(f'intervalo_dos_estocasticos {intervalo_dos_estocasticos}')
            if i >= intervalo_dos_estocasticos:
                preditc_train, preditc_test, preditc_train_best, preditc_test_best = self.method_slector.method_Predict(m,ts,tain_data,test_data,1)
                tempoExec = time.time() - t1
                #self.tempoExecModelos[m+"_best"] = tempoExec
                
                for x in range(len(preditc_test)):
                    if preditc_test[x] < 0:
                        preditc_test[x] = 0
                        
                for x in range(len(preditc_test_best)):
                    if preditc_test_best[x] < 0:
                        preditc_test_best[x] = 0
                
                
                self.tempoExecModelos[m] = tempoExec
                print("Tempo de execução: {} segundos".format(tempoExec))
                self.result[m] = preditc_test
                self.result[m+"_best"] = preditc_test_best
                self.result_train[m] = preditc_train
                self.result_train[m+"_best"] = preditc_train_best
                plt.plot(preditc_test, label=m)
                plt.plot(preditc_test_best, label=m+"_best")
            else:
                preditc_train, preditc_test = self.method_slector.method_Predict(m,ts,tain_data,test_data)
                tempoExec = time.time() - t1
                
                for x in range(len(preditc_test)):
                    if preditc_test[x] < 0:
                        preditc_test[x] = 0
                        
                self.tempoExecModelos[m] = tempoExec           
                print("Tempo de execução: {} segundos".format(tempoExec))
                self.result[m] = preditc_test
                self.result_train[m] = preditc_train
                plt.plot(preditc_test, label=m)
            i += 1
        plt.legend(loc="upper left")
        plt.gcf().autofmt_xdate()
        plt.show()
        
        print(self.result)
        
        print(f" tamanho validação: {tamanhoValidacao}")
        print(f" len(self.result): {len(self.result)}")
        
        result_validation = self.result.copy()
        for m in result_validation:
            result_validation[m] = result_validation[m][:-tamanhoValidacao]
            
        forecast_errors_mse = self.predict_erros.error_MSE(methods,result_validation,test_data[:-tamanhoValidacao])      
        forecast_errors_rmse = self.predict_erros.error_RMSE(methods,result_validation,test_data[:-tamanhoValidacao])
        #train_errors_rmse = self.predict_erros.error_RMSE(methods,self.result_train,tain_data)

        tempo_total = sum(self.tempoExecModelos.values())
        print(f"Tempo Total: {tempo_total}")
        
        t1 = time.time()
        result_comb_media = self.ensembles_strategist.Mean_Combination(data,len(test_data),methods,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_mediana = self.ensembles_strategist.Median_Combination(data,len(test_data),methods,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Mediana'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_media_aparada = self.ensembles_strategist.Trimmed_Mean_Combination(data,len(test_data),methods,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media Aparada'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_media_ponderada = self.ensembles_strategist.weighted_average_Combination(data,len(test_data),methods,self.result,forecast_errors_rmse)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media Ponderada'] = tempoExec + tempo_total
        
        
        print("")
        print("erros:")
        print(forecast_errors_rmse)
        
        self.result['Comb Media'] = result_comb_media
        self.result['Comb Mediana'] = result_comb_mediana
        self.result['Comb Media Aparada'] = result_comb_media_aparada
        self.result['Comb Media Ponderada'] = result_comb_media_ponderada
        
        
        methods_aux = methods[:]
        for m in range(len(methods_aux)):
            if methods_aux[m].find("RNN") >= 0:
                methods_aux[m] = methods_aux[m]+"_best"
                
        print(f"novos m :{methods_aux}")
        time.sleep(5)
        
        result_validation = self.result.copy()
        for m in result_validation:
            result_validation[m] = result_validation[m][:-tamanhoValidacao]
            
        forecast_errors_mse = self.predict_erros.error_MSE(methods_aux,result_validation,test_data[:-tamanhoValidacao])      
        forecast_errors_rmse = self.predict_erros.error_RMSE(methods_aux,result_validation,test_data[:-tamanhoValidacao])
        


        print(f"Tempo Total: {tempo_total}")
        
        t1 = time.time()
        result_comb_media_best = self.ensembles_strategist.Mean_Combination(data,len(test_data),methods_aux,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media best'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_mediana_best = self.ensembles_strategist.Median_Combination(data,len(test_data),methods_aux,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Mediana best'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_media_aparada_best = self.ensembles_strategist.Trimmed_Mean_Combination(data,len(test_data),methods_aux,self.result)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media Aparada best'] = tempoExec + tempo_total
        
        t1 = time.time()
        result_comb_media_ponderada_best = self.ensembles_strategist.weighted_average_Combination(data,len(test_data),methods_aux,self.result,forecast_errors_mse)
        tempoExec = time.time() - t1
        self.tempoExecModelos['Comb Media Ponderada best'] = tempoExec + tempo_total
        
        
        self.result['Comb Media best'] = result_comb_media_best
        self.result['Comb Mediana best'] = result_comb_mediana_best
        self.result['Comb Media Aparada best'] = result_comb_media_aparada_best
        self.result['Comb Media Ponderada best'] = result_comb_media_ponderada_best
        
        plt.plot(data,label='Original')
        plt.plot(result_comb_media, label='Comb Media')
        plt.plot(result_comb_mediana, label='Comb Mediana')   
        plt.plot(result_comb_media_aparada, label='Comb Media Aparada')   
        plt.plot(result_comb_media_ponderada, label='Comb Media Ponderada')   
        plt.plot(result_comb_media_best, label='Comb Media best')
        plt.plot(result_comb_mediana_best, label='Comb Mediana best')   
        plt.plot(result_comb_media_aparada_best, label='Comb Media Aparada best')   
        plt.plot(result_comb_media_ponderada_best, label='Comb Media Ponderada best') 
        plt.legend(loc="upper left")
        plt.gcf().autofmt_xdate()
        plt.show()
        
        return self.result, self.result_train, self.tempoExecModelos
        

    