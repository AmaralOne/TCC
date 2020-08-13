# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:10:42 2020

@author: Amaral
"""


from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import GridSearchCV
from Utils_neural_network import Utils_neural_network as Util_NN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import UtilsM3
import autoSVR
#Carregar uma Série Temporal do Conjunto M3
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1681")
dataset = ts
dataset = np.array(dataset)
dataset = dataset.reshape(-1,1)
scaler_x = StandardScaler().fit(dataset)
dataset = scaler_x.transform(dataset)
__train, __test = Util_NN.split_train_and_test(dataset,0.8,2,18)
__trainX, __trainY = Util_NN.create_dataset(__train, 2)
__testX, __testY = Util_NN.create_dataset(__test, 2)  
x1 = __trainX
x2 = __trainY


x1 = x1.reshape(-1,2)

x2 = x2.reshape(-1)

__trainX = __trainX.reshape(-1,1)

__trainY = __trainY.reshape(-1)

n_samples, n_features = 1000, 50
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
svr = SVR()

clf = GridSearchCV(svr, parameters, scoring = 'neg_root_mean_squared_error', verbose = 1)
print("Entrou 2 x:",len(X)," y:",len(y))

clf.fit(__trainX,__trainY)

#best score
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
clf = SVR(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], epsilon=best_parameters['epsilon'])
clf.fit(__trainX,__trainY)

pred_y_train = clf.predict(x1)

pred_y_train = scaler_x.inverse_transform(pred_y_train)


plt.plot(ts.values)
plt.plot(pred_y_train)
plt.title('SVR')
plt.show()


plt.plot(ts.values)
plt.title('SVR')
plt.show()
ts = U_m3.buildM3DataFrame("N1681")
model = autoSVR.autoSVR()
model.prepararSerie(ts,18,1,6)
preditc_train, preditc_test = model.fit()

        
plt.plot(ts)
plt.plot(preditc_train)
plt.plot(preditc_test)
plt.title('SVR')
plt.show()
