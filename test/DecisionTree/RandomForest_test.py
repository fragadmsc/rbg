import rbg.DecisionTree.RandomForest as RdForest
import rbg.DecisionTree.RandomForestRegression as RdForestReg
import numpy as np
import pandas as pd

# para impurity: Gini, classification (por padrão), entropy
# m representa o tamanho das divisões de features =============== talvez de pra pegar automaticamente, ou os dois 
features = 100
RdForest.set_params(impurity = 'Gini', m = features/3,n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForest.get_params()

RdForest.fit(X,Y)
RdForest.predict(Data)


#regressão, igualmente
features = 100
RdForestReg.set_params(m = features/3, n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForestReg.get_params()

RdForestReg.fit(X,Y)
RdForestReg.predict(Data)