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

#import pymysql


        
class Individuo():
    def __init__(self, modelos, ts, test_data, result, geracao=0):
        self.ts = ts
        self.test_data = test_data
        self.modelos = modelos
        self.result_predict = result
        self.nota_avaliacao = 0
        self.espaco_usado = 0
        self.erro = 1000000
        self.geracao = geracao
        self.cromossomo = []
        self.ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
        self.predict_erros = Predicts_Erro.Predicts_Erro()
        
        for i in range(len(modelos)+2):
            if random() < 0.5:
                self.cromossomo.append("0")
            else:
                self.cromossomo.append("1")
            
        
    def avaliacao(self):
        methods_selecionados = []
        tamanho_cromossomo = len(self.cromossomo)
        if '1' in self.cromossomo[:-2]:
            for i in range(tamanho_cromossomo-2):
               if self.cromossomo[i] == '1':
                   methods_selecionados.append(self.modelos[i])
            print("Metodos Selecionados: ",methods_selecionados)
            if self.cromossomo[tamanho_cromossomo-2] == '0' and self.cromossomo[tamanho_cromossomo-1] == '0':
                print("Media")
                result_comb = self.ensembles_strategist.Mean_Combination(self.ts,len(self.test_data),methods_selecionados,self.result_predict)
            elif self.cromossomo[tamanho_cromossomo-2] == '0' and self.cromossomo[tamanho_cromossomo-1] == '1':
                result_comb = self.ensembles_strategist.Median_Combination(self.ts,len(self.test_data),methods_selecionados,self.result_predict)
                print("Mediana")
            elif self.cromossomo[tamanho_cromossomo-2] == '1' and self.cromossomo[tamanho_cromossomo-1] == '0':
                result_comb = self.ensembles_strategist.Trimmed_Mean_Combination(self.ts,len(self.test_data),methods_selecionados,self.result_predict)
                print("Media Aparada")
            elif self.cromossomo[tamanho_cromossomo-2] == '1' and self.cromossomo[tamanho_cromossomo-1] == '1':
                print("Media Ponderada")
                forecast_errors_mse = self.predict_erros.error_MSE(methods_selecionados,self.result_predict,self.test_data)   
                result_comb = self.ensembles_strategist.weighted_average_Combination(self.ts,len(self.test_data),methods_selecionados,self.result_predict,forecast_errors_mse)
                
        #self.erro = sklearn.metrics.mean_squared_error(self.test_data,result_comb) 
            self.erro = ut.rmse(self.test_data,result_comb)
            self.nota_avaliacao = (1/self.erro)
        else:
            self.erro = 100000000
            self.nota_avaliacao = 0
        print(self.erro)
        print(self.nota_avaliacao)
        
    def crossover(self, outro_individuo):
        corte = round(random()  * len(self.cromossomo))
        
        filho1 = outro_individuo.cromossomo[0:corte] + self.cromossomo[corte::]
        filho2 = self.cromossomo[0:corte] + outro_individuo.cromossomo[corte::]
        
        filhos = [Individuo(self.modelos, self.ts, self.test_data, self.result_predict, self.geracao + 1),
                  Individuo(self.modelos, self.ts, self.test_data, self.result_predict, self.geracao + 1)]
        filhos[0].cromossomo = filho1
        filhos[1].cromossomo = filho2
        return filhos
    
    def mutacao(self, taxa_mutacao):
        #print("Antes %s " % self.cromossomo)
        for i in range(len(self.cromossomo)):
            if random() < taxa_mutacao:
                if self.cromossomo[i] == '1':
                    self.cromossomo[i] = '0'
                else:
                    self.cromossomo[i] = '1'
        #print("Depois %s " % self.cromossomo)
        return self
        
class AlgoritmoGenetico():
    def __init__(self, tamanho_populacao):
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        self.geracao = 0
        self.melhor_solucao = 0
        self.lista_solucoes = []
        
    def inicializa_populacao(self, modelos, ts, test_data, result):
        for i in range(self.tamanho_populacao):
            self.populacao.append(Individuo(modelos, ts, test_data, result))
        self.melhor_solucao = self.populacao[0]
        
    def ordena_populacao(self):
        self.populacao = sorted(self.populacao,
                                key = lambda populacao: populacao.nota_avaliacao,
                                reverse = True)
        
    def melhor_individuo(self, individuo):
        if individuo.nota_avaliacao > self.melhor_solucao.nota_avaliacao:
            self.melhor_solucao = individuo
            
    def soma_avaliacoes(self):
        soma = 0
        for individuo in self.populacao:
           soma += individuo.nota_avaliacao
        return soma
    
    def seleciona_pai(self, soma_avaliacao):
        pai = -1
        valor_sorteado = random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].nota_avaliacao
            pai += 1
            i += 1
        return pai
    
    def visualiza_geracao(self):
        melhor = self.populacao[0]
        print("G:%s -> Valor: %s Espaço: %s Cromossomo: %s" % (self.populacao[0].geracao,
                                                               melhor.nota_avaliacao,
                                                               melhor.erro,
                                                               melhor.cromossomo))
    
    def resolver(self, taxa_mutacao, numero_geracoes, modelos, ts, test_data, result):
        self.inicializa_populacao(modelos, ts, test_data, result)
        
        for individuo in self.populacao:
            individuo.avaliacao()
        
        self.ordena_populacao()
        self.melhor_solucao = self.populacao[0]
        self.lista_solucoes.append(self.melhor_solucao.nota_avaliacao)
        
        metade = self.tamanho_populacao // 2
        
        for i in range(metade):
            self.populacao.pop()
        
        self.visualiza_geracao()
        
        for geracao in range(numero_geracoes):
            soma_avaliacao = self.soma_avaliacoes()
            nova_populacao = []
            
            for individuos_gerados in range(0, (self.tamanho_populacao//2), 2):
                pai1 = self.seleciona_pai(soma_avaliacao)
                pai2 = self.seleciona_pai(soma_avaliacao)
                
                filhos = self.populacao[pai1].crossover(self.populacao[pai2])
                
                nova_populacao.append(filhos[0].mutacao(taxa_mutacao))
                nova_populacao.append(filhos[1].mutacao(taxa_mutacao))
            
            self.populacao.extend(list(nova_populacao))
            
            for individuo in self.populacao:
                individuo.avaliacao()
            
            self.ordena_populacao()
            
            metade = self.tamanho_populacao // 2
        
            for i in range(metade):
                self.populacao.pop()
            
            self.visualiza_geracao()
            
            melhor = self.populacao[0]
            self.lista_solucoes.append(melhor.nota_avaliacao)
            self.melhor_individuo(melhor)
        
        print("\nMelhor solução -> G: %s Valor: %s Espaço: %s Cromossomo: %s" %
              (self.melhor_solucao.geracao,
               self.melhor_solucao.nota_avaliacao,
               self.melhor_solucao.erro,
               self.melhor_solucao.cromossomo))
        
        return self.melhor_solucao.cromossomo
        
        
if __name__ == '__main__':
    
    cif = UtilsCIF.UtilsCIF()
    index = cif.listarIndex()
    
    cols = ['serie','smape_mean','smape_std', 'rmse_mean','rmse_std']
    
    reuslt_comb_selection = pd.DataFrame(columns=cols)
   
   # modelos = ['ses','naive','holt','Ar','Croston', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
              # 'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
              # 'MLP A4','MLP A5', 'MLP A6','RNN A1','RNN A2','RNN A3',
               #'RNN A4', 'RNN A5','RNN A6', 'ELM']
               
    modelos = ['ses','naive','holt','Ar', 'Arima','SVR A1', 'SVR A2', 'SVR A3',
               'SVR A4', 'SVR A5', 'SVR A6','NNAR','NNAR RNN','MLP A1','MLP A2','MLP A3',
              'RNN A1','RNN A2','RNN A3',
                'ELM']
    

    #index = index[]
    
    for serie in index:
        erros_smape = []
        erros_rmse = []
        for i in range(20):
            #serie = 'ts1'
            #arquivo_result = pd.read_excel("Resultado_Predict_N1679.xlsx")
            arquivo_result = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif\\Resultado_Predict_'+serie+'.xlsx',None)
            
            #results = {}
            #for m in range(len(arquivo_result)):
                #print(arquivo_result.iloc[m][0])
                #results[arquivo_result.iloc[m][0]] = pd.Series(arquivo_result.iloc[m][1:])
                
            #U_m3 = UtilsM3.UtilsM3()
            #index = U_m3.listarIndex()
            #ts = U_m3.buildM3DataFrame("N1679")
            
            #tamanho_teste = 18
            
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
            inico_de_validacao = (tamanho_serie-(tamanho_teste*2))
            trainData = ts[:incio_de_teste]
            testData = ts[incio_de_teste:]
            validationData = ts[inico_de_validacao:incio_de_teste]
            
            
            tamanho_populacao = 40
            taxa_mutacao = 0.05
            numero_geracoes = 100
            ag = AlgoritmoGenetico(tamanho_populacao)
            resultado = ag.resolver(taxa_mutacao, numero_geracoes,  modelos, ts[:], validationData, results)
            modelos_escolhidos = []
            for i in range(len(resultado)-2):
                if resultado[i] == '1':
                    print(modelos[i])
                    modelos_escolhidos.append(modelos[i])
                    
            
            
            #for valor in ag.lista_solucoes:
            #    print(valor)
            plt.plot(ag.lista_solucoes)
            plt.title("Acompanhamento dos valores")
            plt.show() 
            
            result_series = pd.read_excel('C:\\Users\\Amaral\\Documents\\Faculdade\\tcc\\seletor de Modelo\Resut_cif_Retreino\\Resultado_Predict_retreino_'+serie+'.xlsx',None)   
            result_prediction = result_series.pop('predict_test')
            
            ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
            predict_erros = Predicts_Erro.Predicts_Erro()
            results_t = {}
            for m in range(len(result_prediction)):
                if result_prediction.iloc[m][0] not in modelos_escolhidos:
                    continue           
                results_t[result_prediction.iloc[m][0]] = pd.Series(result_prediction.iloc[m][1:])
            
            tamanho_results = len(resultado)
            if resultado[tamanho_results-2] == '0' and resultado[tamanho_results-1] == '0':
                result_comb = ensembles_strategist.Mean_Combination(ts,len(validationData),modelos_escolhidos,results_t)
                print("Media")
            elif resultado[tamanho_results-2] == '0' and resultado[tamanho_results-1] == '1':
                result_comb = ensembles_strategist.Median_Combination(ts,len(validationData),modelos_escolhidos,results_t)       
                print("Mediana")
            elif resultado[tamanho_results-2] == '1' and resultado[tamanho_results-1] == '0':
                result_comb = ensembles_strategist.Trimmed_Mean_Combination(ts,len(validationData),modelos_escolhidos,results_t) 
                print("Média aparada")        
            elif resultado[tamanho_results-2] == '1' and resultado[tamanho_results-1] == '1':
                forecast_errors_mse = predict_erros.error_MSE(modelos_escolhidos,results,validationData) 
                result_comb = ensembles_strategist.weighted_average_Combination(ts,len(validationData),modelos_escolhidos,results_t,forecast_errors_mse)               
                print("Média Ponderada")
                
            plt.plot(ts)
            plt.plot(result_comb)
            plt.title("Resultado comb selection")
            plt.show() 
            
            forecast_errors_smape = ut.smape(testData,result_comb)
            forecast_errors_rmse = ut.rmse(testData,result_comb)
        
            erros_smape.append(forecast_errors_smape)
            erros_rmse.append(forecast_errors_rmse)
            print(f"smape: {forecast_errors_smape}")
            print(f"smape: {forecast_errors_rmse}")
            
        erros_smape = np.array(erros_smape)
        erros_rmse = np.array(erros_rmse)  
        line = {'serie':serie,
                 'smape_mean': round(erros_smape.mean(),3),
                 'smape_std': round(erros_smape.std(),3),
                 'rmse_mean': round(erros_rmse.mean(),3),
                 'rmse_std': round(erros_rmse.std(),3),
                 }
        reuslt_comb_selection = reuslt_comb_selection.append(line,ignore_index=True)
    reuslt_comb_selection.to_excel(excel_writer='Resultado_Ensemble.xlsx',index=False)
    