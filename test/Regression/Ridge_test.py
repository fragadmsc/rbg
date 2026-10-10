import rbg.Regression.Ridge as Ridge
import numpy as np
import pandas as pd
import pytest


#verificar apenas se aceita receber o parametro alpha
Ridge.set_params(alpha = 1,intecept = False)
Ridge.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Ridge.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Ridge.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Ridge.coef

#verifica se retorna um numpy float
Ridge.score()