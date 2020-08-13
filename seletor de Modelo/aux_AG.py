# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:52:54 2020

@author: Amaral


"""


U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1679")

tamanho_teste = 18
    
#Dividir a Série Temporal em treino e Teste
tamanho_serie = len(ts)
incio_de_teste = (tamanho_serie-tamanho_teste)
trainData = ts[:incio_de_teste]
testData = ts[incio_de_teste:]

methods = ['Arima','SVR A1', 'NNAR']
ensembles_strategist = Ensemble_Strategist.Ensemble_Strategist()
predict_erros = Predicts_Erro.Predicts_Erro()

result_comb_media = ensembles_strategist.Mean_Combination(ts,len(testData),methods,results)

plt.plot(ts,label='Original')
plt.plot(result_comb_media, label='Comb Media')
plt.legend(loc="upper left")
plt.gcf().autofmt_xdate()
plt.show()
        
