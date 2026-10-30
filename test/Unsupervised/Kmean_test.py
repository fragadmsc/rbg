from rbg.Unsupervised.KMeans import KMeans
import numpy as np
import pytest

'''

#testa se consegue receber os parametros e se retorna o esperado
KMean.set_params(k = 10, centroid_selection = 'Kmean++', max_iter = 100)

#testa se consegue receber esses parametros tmabém
KMean.set_params(k= 10, centroid_selection = [(10.5,11.4),(12.0,1.4),(90.0,43.12)])


#retorna um dicionario com key:nome do parametr e valor: valor do parametro
KMean.get_params()

#testa se funciona
KMean.fit(X)

#testa se retona o que precisa retornar, uma array com tamanho de linhas igual ao de x, mas com 1 coluna apenas
KMean.clusters

#retorna a coordenada dos centroides, ou seja, uma matriz com k linhas e o numero de colunas de X
KMean.centroids

'''


class TestKMean:
    
    def test_params(self):
        modelo_kmean = KMeans(k=10, centroid_selection='Kmeans++', max_iter=100)
        assert modelo_kmean.k == 10
        assert modelo_kmean.centroid_selection == 'Kmeans++'
        assert modelo_kmean.max_iter == 100

        centroids = [(10.5, 11.4), (12.0, 1.4), (90.0, 43.12)]
        modelo_kmean = KMeans(k=3, centroid_selection=centroids)
        assert modelo_kmean.centroid_selection == centroids

    def test_fit_and_attributes(self):
        X = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        
        n_amostras = X.shape[0]
        n_features = X.shape[1]

        k = 2
        modelo_kmean = KMeans(k=k, centroid_selection='Kmeans++', max_iter=100)
        
        modelo_kmean.fit(X)
        assert isinstance(modelo_kmean.clusters, np.ndarray)
        assert modelo_kmean.clusters.shape == (n_amostras, )

        assert isinstance(modelo_kmean.centroids, np.ndarray)
        assert modelo_kmean.centroids.shape == (k, n_features)

        for centroid in modelo_kmean.centroids:
            assert len(centroid) == n_features
