# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:51:47 2020

@author: Amaral
"""
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
import mlrose
import seaborn as sns
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
            if solucao[m] == '1':
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
            if solucao[i] == '1':
                n += 1
                methods_selecionados.append(modelos[i])
        #print("Metodos Selecionados: ",methods_selecionados)
                
        if (solucao[tamanho_cromossomo-2] == '0') and solucao[tamanho_cromossomo-1] == '0':
            #print("Media")
            result_comb = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
        elif (solucao[tamanho_cromossomo-2] == '0') and solucao[tamanho_cromossomo-1] == '1':
            result_comb = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
            #print("Mediana")
        elif solucao[tamanho_cromossomo-2] == '1' and solucao[tamanho_cromossomo-1] == '0':
            result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results)
            #print("Media Aparada")
        elif solucao[tamanho_cromossomo-2] == '1' and solucao[tamanho_cromossomo-1] == '1':
            #print("Media Ponderada")
            forecast_errors_mse = predict_erros.error_RMSE(methods_selecionados,results,validationData)   
            result_comb = ensembles_strategist.weighted_average_Combination(ts[:-tamanho_teste],len(validationData),methods_selecionados,results,forecast_errors_mse)
                        
        erro = ut.rmse(validationData,result_comb)
        #nota_avaliacao = (1/erro) - ((1/erro)*0.02*n)
        nota_avaliacao = (1/erro)
        return nota_avaliacao
    else:
        erro = 100000000
        nota_avaliacao = 0
        return 0
    print(erro)
    print(nota_avaliacao)        
    
def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def transformaStringSolucao(x):
    s = ''    
    for c in x:
        s =  s + c
    return s

def annealing(random_start,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000,
              debug=True,tam = 8,h = 3):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = random_start(tam)
    cost = cost_function(state)
    states, costs = [state], [cost]
    trocou = 0
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_state = random_neighbour(state, h)
        new_cost = cost_function(new_state)
        if debug: print('Step #{:3}/{:2} : T = {:3}, state = {:3}, cost = {:3}, new_state = {:3}, new_cost = {:3} '.format(step, maxsteps, T, transformaStringSolucao(state), cost, transformaStringSolucao(new_state), new_cost))
        if ((acceptance_probability(cost, new_cost, T) > rn.random())):
            print('trocou')
            trocou = trocou + 1
            print(trocou)
            print()
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
    return state, cost_function(state), states, costs

def annealing2(random_start,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000,
              debug=True,tam = 8,h = 3,temp_min = 10**-20, temp_inicial = 1000, tentativas = 2,alfa = 0.90):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = random_start(tam)
    cost = cost_function(state)
    states, costs = [state], [cost]
    trocou = 0
    T = temp_inicial
    steps = 0
    while(T > temp_min):
        for repita in range(tentativas):
            new_state = random_neighbour(state, h)
            new_cost = cost_function(new_state)
            if debug: print('Step #{:3}/{:2} : T = {:3}, state = {:3}, cost = {:3}, new_state = {:3}, new_cost = {:3} '.format(steps, maxsteps, T, transformaStringSolucao(state), cost, transformaStringSolucao(new_state), new_cost))
            if ((acceptance_probability(cost, new_cost, T) > rn.random())):
                #print('trocou')
                trocou = trocou + 1
                #print(trocou)
                #print()
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
        T = T * alfa
        steps = steps + 1
        if(steps == maxsteps):
            break

            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
    return state, cost_function(state), states, costs

def random_start(tam):
    """ Random point in the interval."""
    cromossomo = []
    for i in range(tam):
        if random() < 0.5:
            cromossomo.append("0")
        else:
            cromossomo.append("1")   
    return cromossomo



def random_neighbour(x, fraction=1):
    x = transformaStringSolucao(x)
    fraction = int(fraction)
    h_s = 0
    s = ''
    #print(x)
    tam = len(x)
    #print('tam: ', tam)
    #print('fraction: ',fraction)
    assert fraction <= tam
    while(h_s != fraction):
        cromossomo = []
        for i in range(tam):
            if random() < 0.5:
                cromossomo.append("0")
            else:
                cromossomo.append("1")        
        s = ''    
        for c in cromossomo:
            s =  s + c
    
        h_s = hamming_distance(s,x)
    novo = []
    for i in s:
        novo.append(i)
    #print('novo: ',novo)
    return novo

def acceptance_probability(cost, new_cost, temperature):
    if new_cost > cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (cost - new_cost ) / temperature)
        #print('p : ',p)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p
    
def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))


cif = UtilsCIF.UtilsCIF()
index = cif.listarIndex()
cols = ['serie','smape_mean','smape_std', 'rmse_mean','rmse_std']
    
media_ponderada_result = []
reuslt_comb_selection = pd.DataFrame(columns=cols)


cols_r = ['erro validation', 'erro test','estrategia comb','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
for serie in index:
    #serie = 'ts22'
    erros_smape = []
    erros_rmse = []
    modelos_selecionados_ensemble = pd.DataFrame(columns=cols_r)
    for i in range(20):

        arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_validacao30Porcent_'+serie+'.xlsx',None)
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
            
        #Cenario 16 SA com mudança na fittness
        init_temp=1000
        decay=0.90
        min_temp=10**-20
        max_attempts=2
        
        #Cenario 17 SA com mudança na fittness
        init_temp=1000
        decay=0.85
        min_temp=10**-20
        max_attempts=2
        
        #Cenario 18 SA com mudança na fittness
        init_temp=1000
        decay=0.95
        min_temp=10**-20
        max_attempts=2

        #state, c, states, costs = annealing(random_start, avaliacao, random_neighbour, acceptance_probability, temperature, maxsteps=500, debug=True,tam = len(modelos)+2,h = 3);
        state, c, states, costs = annealing2(random_start, avaliacao, random_neighbour, acceptance_probability, temperature, maxsteps=2000, debug=False,tam = len(modelos)+2,h = 3, temp_min= min_temp,temp_inicial = init_temp,tentativas = max_attempts,alfa = decay);
        
        
        #Mostra o melhor resultado


        plt.figure()
        plt.suptitle("Evolution of states and costs of the simulated annealing")
        plt.plot(costs, 'b')
        plt.title("Costs")
        plt.show()
        

        resultado = state
        modelos_escolhidos = []
        for i in range(len(resultado)-2):
            if resultado[i] == '1':
                print(modelos[i])
                modelos_escolhidos.append(modelos[i])
                    
            
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
        if resultado[tamanho_results-2] == '0' and resultado[tamanho_results-1] == '0':
            result_comb_v = ensembles_strategist.Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            result_comb = ensembles_strategist.Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
            print("Media")
            estrategia_comb = "Media"
        elif resultado[tamanho_results-2] == '0' and resultado[tamanho_results-1] == '1':
            result_comb_v = ensembles_strategist.Median_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            result_comb = ensembles_strategist.Median_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)       
            print("Mediana")
            estrategia_comb = "Mediana"
        elif resultado[tamanho_results-2] == '1' and resultado[tamanho_results-1] == '0':
            result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts,tamanho_teste,modelos_escolhidos,results_t)
            result_comb_v = ensembles_strategist.Trimmed_Mean_Combination(ts[:-tamanho_teste],len(validationData),modelos_escolhidos,results)
            print("Média aparada")   
            estrategia_comb = "Média aparada"
        elif resultado[tamanho_results-2] == '1' and resultado[tamanho_results-1] == '1':
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
    modelos_selecionados_ensemble.to_excel(excel_writer='ResultadoEnsemble/mlrose17SA_cenario_Ensemble_'+serie+'.xlsx',index=False)
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
reuslt_comb_selection.to_excel(excel_writer='Resultado_Ensemble_smape_cenario_teste_mlrose18sa.xlsx',index=False)
    


    
    