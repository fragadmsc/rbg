import rbg.DecisionTree.Bagging as Bagging
import rbg.DecisionTree.BaggingRegression as BaggingReg
import numpy as np
import pandas as pd

# para impurity: Gini, classification (por padrão), entropy
features = 100
Bagging.set_params(impurity = 'Gini', n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
Bagging.get_params()

Bagging.fit(X,Y)
Bagging.predict(Data)


#regressão, igualmente
features = 100
BaggingReg.set_params(n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
BaggingReg.get_params()

BaggingReg.fit(X,Y)
BaggingReg.predict(Data)