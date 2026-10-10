import rbg.Regression.Lasso as Lasso
import numpy as np
import pandas as pd
import pytest


#verificar apenas se aceita receber o parametro alpha
Lasso.set_params(alpha = 1,intercept = False)
Lasso.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Lasso.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Lasso.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Lasso.coef

#verifica se retorna um numpy float
Lasso.score()