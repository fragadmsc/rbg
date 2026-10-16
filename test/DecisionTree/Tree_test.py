import rbg.DecisionTree.ClassificationTree as Classification
import rbg.DecisionTree.RegressionTree as Regression
import numpy as np
import pandas as pd

# para impurity: Gini, classification (por padrão), entropy
features = 100
Classification.set_params(impurity = 'Gini', min_inpurity_decrease = 0.1 , max_depth = 30)
Classification.get_params()

Classification.fit(X,Y)
Classification.predict(Data)


#regressão, igualmente
features = 100
Regression.set_params(min_inpurity_decrease = 0.1 , max_depth = 30)
Regression.get_params()

Regression.fit(X,Y)
Regression.predict(Data)