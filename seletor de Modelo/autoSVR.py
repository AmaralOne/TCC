from Utils_neural_network import Utils_neural_network as Util_NN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class autoSVR:

    def __init__(self):  
        print()
        
    def prepararSerie(self,data,horizion=18, frequency = 1, lag = 1):
        self.dataset = data
        
        
        
        self.ts = np.array(data)
        self.ts = self.ts.reshape(-1,1)
        self.lag = lag

        self.scaler_x = StandardScaler().fit(self.ts)

        self.ts = self.scaler_x.transform(self.ts)
        

        
        k = self.scaler_x.inverse_transform(self.ts)
        

       
        self.train, self.test = Util_NN.split_train_and_test(self.ts,0.8,lag,horizion)
        self.trainX, self.trainY = Util_NN.create_dataset(self.train, lag)
        self.testX, self.testY = Util_NN.create_dataset(self.test, lag)  
        
        
    def fit(self):      
        from sklearn.svm import SVR
        import numpy as np
        from sklearn.model_selection import GridSearchCV
        #print('------------')
        #print('training svr model ......')
        
        
        parameters = {"C": [1e-1,1e-2,1e-3,1e1, 1e2, 1e3], "gamma": [0.00025, 0.00020, 0.00015, 0.00010],
              "epsilon": [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]}
        scores = ['precision', 'recall']
        
        self.trainX = self.trainX.reshape(-1,self.lag)
        
        self.testX = self.testX.reshape(-1,self.lag)

        self.trainY =  self.trainY.reshape(-1)
        
        #parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
        svr = SVR()
        
        x1 = self.trainX
        x2 =  self.trainY
        #'neg_root_mean_squared_error'

        clf = GridSearchCV(svr, parameters, scoring ='neg_root_mean_squared_error' , verbose = 0)
        clf.fit(x1, x2)
        
        #best score
        #print("Best score: %0.3f" % clf.best_score_)
        #print("Best parameters set:")
        best_parameters = clf.best_estimator_.get_params()
        #for param_name in sorted(parameters.keys()):
         #   print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        clf = SVR(best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], epsilon=best_parameters['epsilon'])
        clf.fit(self.trainX, self.trainY)

        pred_y_train = clf.predict(self.trainX)
        pred_y_test = clf.predict(self.testX)
        

        
        pred_y_train = self.scaler_x.inverse_transform(pred_y_train)
        

        
        pred_y_test = self.scaler_x.inverse_transform(pred_y_test)
        
        
        pred_y_train = pd.Series(pred_y_train,self.dataset.index[self.lag:len(pred_y_train)+self.lag])
    
        pred_y_test = np.array(pred_y_test)
        pred_y_test = pred_y_test.reshape((len(self.dataset)-len(pred_y_train)-(self.lag)))
        
        pred_y_test = pd.Series(pred_y_test,self.dataset.index[len(pred_y_train)+(self.lag):len(self.dataset)])
    
    
             
        return pred_y_train, pred_y_test
    
    def teste(self):
        from sklearn.svm import SVR
        import numpy as np
        from sklearn.model_selection import GridSearchCV
        n_samples, n_features = 10, 5
        np.random.seed(0)
        y = np.random.randn(n_samples)
        X = np.random.randn(n_samples, n_features)
        parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
        svr = SVR()
        print("Entrou 1")
        clf = GridSearchCV(svr, parameters,verbose = 0)
        print("Entrou 2 x:",len(self.trainX)," y:",len(self.trainY))
        clf.fit(self.trainX, self.trainY)
        print("Entrou 3")
        print(clf.best_params_)
    
    def lag(self):
        return True
             
        

