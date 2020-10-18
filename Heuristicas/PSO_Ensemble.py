# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:51:47 2020

@author: Amaral
"""
from __future__ import division
from __future__ import print_function, division  # Python 2 compatibility if needed
from random import random
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl
from scipy import optimize       # to compare
from random import random
import matplotlib.pyplot as plt
import Ensemble_Strategist
import Predicts_Erro
import sklearn
import pandas as pd
import UtilsM3
import UtilsCIF
from util import Utils as ut
import numpy as np
import time
#import mlrose
import seaborn as sns

import matplotlib.pyplot as plt
import random
import math
import numpy as np
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

FIGSIZE = (19, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE

modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
              'RNN A1','RNN A2','RNN A3',
                'ELM']


ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
predict_erros = Predicts_Erro.Predicts_Erro()

def verificaSeTemDoisModelos(tamanho,solucao):
        k = 0
        for m in range(tamanho):
            if solucao[m] == 1:
                k = k + 1
            if k == 2:
                return True
        return False
    
def avaliacao(solucao):

    methods_selecionados = []
    tamanho_cromossomo = len(solucao)
    n =0 
    if verificaSeTemDoisModelos(tamanho_cromossomo-2,solucao) == True:
        for i in range(tamanho_cromossomo-2):
            if solucao[i] == 1:
                n += 1
                methods_selecionados.append(modelos[i])
        #print("Metodos Selecionados: ",methods_selecionados)
        melhor = 9000000000
            
        # result_comb = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
        # erro_aux = ut.rmse(validationData,result_comb)
        # if erro_aux < melhor:
        #     melhor = erro_aux
        #     result_comb_f = result_comb
        #     solucao[tamanho_cromossomo-2] = 0
        #     solucao[tamanho_cromossomo-1] = 0
                
        # result_comb = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
        # erro_aux = ut.rmse(validationData,result_comb)
        # if erro_aux < melhor:
        #     melhor = erro_aux
        #     result_comb_f = result_comb
        #     solucao[tamanho_cromossomo-2] = 0
        #     solucao[tamanho_cromossomo-1] = 1
                
        # result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
        # erro_aux = ut.rmse(validationData,result_comb)
        # if erro_aux < melhor:
        #     melhor = erro_aux
        #     result_comb_f = result_comb
        #     solucao[tamanho_cromossomo-2] = 1
        #     solucao[tamanho_cromossomo-1] = 0
                
        # forecast_errors_mse = predict_erros.error_RMSE(methods_selecionados,results,validationData)   
        # result_comb = ensembles_strategist.weighted_average_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results,forecast_errors_mse)
        # erro_aux = ut.rmse(validationData,result_comb)
        # if erro_aux < melhor:
        #     melhor = erro_aux
        #     result_comb_f = result_comb
        #     solucao[tamanho_cromossomo-2] = 1
        #     solucao[tamanho_cromossomo-1] = 1
                
        if (solucao[tamanho_cromossomo-2] == 0) and solucao[tamanho_cromossomo-1] == 0:
            print("Media")
            result_comb = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
        elif (solucao[tamanho_cromossomo-2] == 0) and solucao[tamanho_cromossomo-1] == 1:
            result_comb = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
            print("Mediana")
        elif solucao[tamanho_cromossomo-2] == 1 and solucao[tamanho_cromossomo-1] == 0:
            result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
            print("Media Aparada")
        elif solucao[tamanho_cromossomo-2] == 1 and solucao[tamanho_cromossomo-1] == 1:
            print("Media Ponderada")
            forecast_errors_mse = predict_erros.error_RMSE(methods_selecionados,results,validationData)   
            result_comb = ensembles_strategist.weighted_average_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results,forecast_errors_mse)
                        
        # erro = ut.rmse(validationData,result_comb)
        erro = ut.smape(validationData,result_comb)
        #nota_avaliacao = (1/erro) - ((1/erro)*0.02*n)
        nota_avaliacao = (1/erro)
        return nota_avaliacao
    else:
        erro = 100000000
        nota_avaliacao = 0  
        return 0
    print(erro)
    print(nota_avaliacao)        
    
#--- MAIN 
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(0,1))
            self.position_i.append(random.randint(0, 1))

    # evaluate current fitness
    def evaluate(self,costFunc):
        print(f'particula: {self.position_i}')
        self.err_i=costFunc(self.position_i)
        print(f"erro: {self.err_i}")

        # check to see if the current position is an individual best
        if self.err_i > self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0,num_dimensions):
            #self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            self.position_i[i] = self._compute_position(self.velocity_i[i])


            # adjust maximum position if necessary
           # if self.position_i[i]>bounds[i][1]:
            #    self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            #if self.position_i[i] < bounds[i][0]:
             #   self.position_i[i]=bounds[i][0]
                
    def _compute_position(self, velocity):
        """Update the position matrix of the swarm

        This computes the next position in a binary swarm. It compares the
        sigmoid output of the velocity-matrix and compares it with a randomly
        generated matrix.

        Parameters
        ----------
        swarm: pyswarms.backend.swarms.Swarm
            a Swarm class
            """
        #print(f'velocidade: {velocity}')
        return (
            np.random.random_sample(size=1)
            < self._sigmoid(velocity)
        ) * 1

    def _sigmoid(self, x):
        """Helper method for the sigmoid function
    
            Parameters
            ----------
            x : numpy.ndarray
                Input vector for sigmoid computation
    
            Returns
            -------
            numpy.ndarray
                Output sigmoid computation
        """
        return 1 / (1 + np.exp(-x))
                
class PSO():
    def __init__(self,costFunc,x0,num_particles,maxiter):
        global num_dimensions
        self.result = []
        self.custo = 0
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group
        acompanha_funcao_fitnnes = []
        acompanha_particula = []
        acompanha_funcao_por_particual = []
        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i > err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
                acompanha_funcao_fitnnes.append(err_best_g)
                acompanha_particula.append(j)
                acompanha_funcao_por_particual.append(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position()
            i+=1

        # print final results
        print ('FINAL:')
        print (pos_best_g)
        print (err_best_g)
        
        #for valor in ag.lista_solucoes:
        #    print(valor)
        plt.plot(acompanha_funcao_fitnnes)
        plt.title("Acompanhamento dos valores")
        plt.show() 
        
        self.result = pos_best_g
        self.custo = err_best_g
    def Resultado(self):
        return self.result, self.custo     

cif = UtilsCIF.UtilsCIF()
index = cif.listarIndex()
cols = ['serie','smape_mean','smape_std', 'rmse_mean','rmse_std']

index = index[:]
media_ponderada_result = []
reuslt_comb_selection = pd.DataFrame(columns=cols)


cols_r = ['erro validation', 'erro test','estrategia comb','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
for serie in index:
    #serie = 'ts22'
    erros_smape = []
    erros_rmse = []
    modelos_selecionados_ensemble = pd.DataFrame(columns=cols_r)
    for i in range(20):

        #arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_validacao30Porcent_'+serie+'.xlsx',None)
        #arquivo_result = pd.read_excel('D:\TCC\\seletor de Modelo\\Resut_cif\\Resultado_Predict_novo_20%'+serie+'.xlsx',None)
        arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_novo_20%'+serie+'.xlsx',None)
        result_validation = arquivo_result.pop('predict_validation')
                   
        results = {}
        for m in range(len(result_validation)):
            if result_validation.iloc[m][0] not in modelos:
                continue           
            results[result_validation.iloc[m][0]] = pd.Series(result_validation.iloc[m][1:])
            
            
        ts, tamanho_teste = cif.serie(serie)
                
        #Dividir a Série Temporal em treino e Teste
        tamanho_serie = len(ts)
        incio_de_teste = (tamanho_serie-tamanho_teste)
        tamanho_validacao = (int)((len(ts)-tamanho_teste)*0.2)
        #tamanho_validacao = (int)((tamanho_teste + (tamanho_teste//2)))
        inico_de_validacao = (tamanho_serie-(tamanho_teste+tamanho_validacao))
        #inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
        trainData = ts[:incio_de_teste]
        testData = ts[incio_de_teste:]
        validationData = ts[inico_de_validacao:incio_de_teste]
            
        #Cenario 1 pso
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant
        
        #Cenario 2 pso
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=2        # cognative constant
        c2=2        # social constant
        
        #Cenario 3 pso, verifica qual melhor estrategia de combinação
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant
        
        
        #Cenario 4 pso, repetir o primeiro cenario
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant

        #Cenario 5 pso, repetir o primeiro cenario
        num_particles=20
        maxiter=50
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant
        
        #Cenario 6 pso, , verifica qual melhor estrategia de combinação
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant
        
        #Cenario 7 pso, repetir o primeiro cenario, com smape na fitness
        num_particles=20
        maxiter=30
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=1        # social constant

        #state, c, states, costs = annealing(random_start, avaliacao, random_neighbour, acceptance_probability, temperature, maxsteps=500, debug=True,tam = len(modelos)+2,h = 3);
        #state, c, states, costs = annealing2(random_start, avaliacao, random_neighbour, acceptance_probability, temperature, maxsteps=2000, debug=False,tam = len(modelos)+2,h = 3, temp_min= min_temp,temp_inicial = init_temp,tentativas = max_attempts,alfa = decay);
        initial = []
        for i in range(0,len(modelos)+2):
            initial.append(0)
        pso = PSO(avaliacao,initial,num_particles=20,maxiter=30)
        pos_best_g, err_best_g = pso.Resultado()
        
        #Mostra o melhor resultado

        

        resultado = pos_best_g
        modelos_escolhidos = []
        for i in range(len(resultado)-2):
            if resultado[i] == 1:
                print(modelos[i])
                modelos_escolhidos.append(modelos[i])
                    
            
        #result_series = pd.read_excel('D:\\TCC\\seletor de Modelo\\Resut_cif_Retreino\\Resultado_Predict_retreino_novo_'+serie+'.xlsx',None)   
        result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif_Retreino\\Resultado_Predict_retreino_novo_'+serie+'.xlsx',None)   
        result_prediction = result_series.pop('predict_test')
            
            
        ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
        predict_erros = Predicts_Erro.Predicts_Erro()
        results_t = {}
        for m in range(len(result_prediction)):
            if result_prediction.iloc[m][0] not in modelos_escolhidos:
                continue           
            results_t[result_prediction.iloc[m][0]] = pd.Series(result_prediction.iloc[m][1:])
        estrategia_comb = ""
        tamanho_results = len(resultado)
        if resultado[tamanho_results-2] == 0 and resultado[tamanho_results-1] == 0:
            result_comb_v = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            result_comb = ensembles_strategist.Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
            print("Media")
            estrategia_comb = "Media"
        elif resultado[tamanho_results-2] == 0 and resultado[tamanho_results-1] == 1:
            result_comb_v = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            result_comb = ensembles_strategist.Median_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)       
            print("Mediana")
            estrategia_comb = "Mediana"
        elif resultado[tamanho_results-2] == 1 and resultado[tamanho_results-1] == 0:
            result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
            result_comb_v = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            print("Média aparada")   
            estrategia_comb = "Média aparada"
        elif resultado[tamanho_results-2] == 1 and resultado[tamanho_results-1] == 1:
            forecast_errors_mse = predict_erros.error_RMSE(modelos_escolhidos,results,validationData) 
            result_comb = ensembles_strategist.weighted_average_Combination(ts,tamanho_teste,modelos_escolhidos,results_t,forecast_errors_mse)    
            result_comb_v = ensembles_strategist.weighted_average_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results,forecast_errors_mse)
            print("Média Ponderada")
            estrategia_comb = "Média Ponderada"
                
                
            
            
        plt.plot(ts)
        plt.plot(result_comb_v)
        plt.plot(result_comb)
        plt.title("Resultado comb selection "+serie)
        plt.show() 
            
        forecast_errors_smape_validation = ut.smape(validationData,result_comb_v)
        forecast_errors_smape = ut.smape(testData,result_comb)
        forecast_errors_rmse = ut.rmse(testData,result_comb)
        
        erros_smape.append(forecast_errors_smape)
        erros_rmse.append(forecast_errors_rmse)
        print(f"smape: {forecast_errors_smape}")
        print(f"smape: {forecast_errors_rmse}")
            
        line_r = {'erro validation': round(forecast_errors_smape_validation,3),
                 'erro test': round(forecast_errors_smape,3),
                 'estrategia comb':estrategia_comb}
        for m in range(len(modelos_escolhidos)):
            line_r[str(m+1)] = modelos_escolhidos[m]

        modelos_selecionados_ensemble = modelos_selecionados_ensemble.append(line_r,ignore_index=True)
    modelos_selecionados_ensemble.to_excel(excel_writer='ResultadoEnsemble/PSO7_cenario_Ensemble_'+serie+'.xlsx',index=False)
    erros_smape = np.array(erros_smape)
    erros_rmse = np.array(erros_rmse)  
    line = {'serie':serie,
                 'smape_mean': round(erros_smape.mean(),3),
                 'smape_std': round(erros_smape.std(),3),
                 'rmse_mean': round(erros_rmse.mean(),3),
                 'rmse_std': round(erros_rmse.std(),3),
                 }
    reuslt_comb_selection = reuslt_comb_selection.append(line,ignore_index=True)
        
        
    results_tt = {}
    for m in range(len(result_prediction)):
        if result_prediction.iloc[m][0] not in modelos:
            continue           
        results_tt[result_prediction.iloc[m][0]] = pd.Series(result_prediction.iloc[m][1:])
        
    forecast_errors_mse = predict_erros.error_RMSE(modelos,results,validationData) 
    result_media_ponderada = ensembles_strategist.weighted_average_Combination(ts,tamanho_teste,modelos,results_tt,forecast_errors_mse)    
    media_ponderada_errors_smape = ut.smape(testData,result_media_ponderada)
    media_ponderada_result.append(media_ponderada_errors_smape)
        
media_ponderada_result = np.array(media_ponderada_result)
print('média ponderada (média): ',media_ponderada_result.mean())
print('média ponderada (std): ',media_ponderada_result.std())
reuslt_comb_selection.to_excel(excel_writer='Resultado_Ensemble_smape_cenario_teste_PSO7.xlsx',index=False)
    


    
    