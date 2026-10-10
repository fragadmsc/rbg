import rbg.Unsupervised.KMean as KMean

#talvez no centroid_selection devemos permitir uma seleção de centroid iniciais como uma array, tipo centroid_selection = [...]
#testa se consegue receber os parametros e se retorna o esperado
KMean.set_params(k = 10, centroid_selection = 'Kmean++', max_iter = 100)
KMean.get_params()

#testa se funciona
KMean.fit(X)

#testa se retona o que precisa retornar, uma array com tamanho de linhas igual ao de x, mas com 1 coluna apenas
KMean.clusters

#retorna a coordenada dos centroides, ou seja, uma matriz com k linhas e o numero de colunas de X
KMean.centroids

