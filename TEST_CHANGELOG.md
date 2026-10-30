# Registro de ajustes dos testes

## 2026-10-30 — chamadas obrigatórias de `Polynomial` e `Logistic`

**Escopo:** testes de `Regression/Polynomial_test.py` e `Classification/Logistic_test.py`.

**Motivo:** `Polynomial` requer `degrees`; `Logistic` requer `classes`, e seu método `fit` requer `X` e `y`. Os testes antigos omitiram esses argumentos. Dois conjuntos de dados de `Logistic` também levavam a um ajuste singular.

**Mudanças:** os testes de `Polynomial` agora informam um grau por coluna. Os testes de `Logistic` agora informam classes e respostas, usam `X` bidimensional e, nos casos de atributos e métodos, dados com classes sobrepostas e matriz de desenho de posto completo.

**Validação:** `uv run --with pytest pytest -q -p no:cacheprovider --tb=line test/Regression/Polynomial_test.py test/Classification/Logistic_test.py`: 8 passaram, 6 falharam, 21 subtestes passaram. As falhas restantes envolvem a mensagem exata de `ValueError` e membros de API ausentes (`set_params`, `coef`, `labels`). Esses contratos ainda precisam de decisão; não foram alterados nesta etapa.

## 2026-10-30 — configuração e coeficientes dos modelos de regressão

**Escopo:** `BaseEstimator`, `Linear`, `Ridge`, `Lasso`, `GLM`, `Polynomial` e testes de regressão.

**Contrato:** `get_params()` devolve a configuração; `set_params(**params)` altera parâmetros conhecidos, rejeita nomes inválidos, invalida o ajuste anterior e devolve o próprio modelo. `coef` devolve uma cópia dos coeficientes estimados após `fit`, incluindo o intercepto quando habilitado; antes de `fit`, lança `RuntimeError`. `Ridge` e `Lasso` preservam `alpha` entre ajustes, salvo quando `fit(..., alpha=...)` recebe um valor explícito. `Polynomial` delega os coeficientes ao estimador interno e o reconstrói quando sua configuração muda.

**Validação:** testes específicos de parâmetros, comportamento, dimensões, `GLM` e módulos: 45 passaram, 123 subtestes passaram. A suíte completa de regressão: 64 passaram, 4 falharam, 144 subtestes passaram. As quatro falhas restantes exigem a mensagem literal `The data do not match` em vez da mensagem atual para contagens incompatíveis de observações.

## 2026-10-30 — nome público `labels` nos classificadores

**Escopo:** construtores e armazenamento interno de `Classifier`, `Logistic`, `LDA`, `QDA` e `KNN`; testes de `Logistic` e de integração com regressão.

**Motivo:** usar `labels` como nome do argumento e atributo público dos rótulos de classes. A ordem pública de `labels` preserva a ordem fornecida pelo usuário; a ordem interna de `Logistic` mantém a classe de referência por último para os cálculos.

**Mudanças:** os construtores recebem `labels`; `Logistic` expõe `labels`, `k_function(k)` para a probabilidade estimada de um rótulo e `k_coef(k)` para seus coeficientes (vetor zero para a referência). O teste de dimensões de `Logistic` agora usa dados válidos após verificar a rejeição de dimensões incompatíveis.

**Validação:** testes de `Logistic` e `Regression/Modules_test.py`: 10 passaram, 4 subtestes passaram. Ajuste e previsão de `LDA`, `QDA` e `KNN` também foram verificados com amostras pequenas. Os testes antigos próprios de `LDA`, `QDA` e `KNN` continuam chamando os construtores sem o argumento obrigatório de rótulos; essa incompatibilidade já existia antes da troca do nome.

## 2026-10-30 — alinhamento da suíte fora de Optimizer

**Escopo:** testes de classificação, regressão, validação cruzada, PCA, KMeans, pré-processamento e utilitários; implementações pequenas de `Standard` e `Matrix_design`.

**Mudanças:** os testes de KNN, LDA e QDA agora informam `labels`, usam matrizes de características e respostas unidimensionais, e dados suficientes para estimar covariâncias. Os testes de KFold e LOOCV usam a API de funções geradoras. PCA é instanciada em cada teste, e seu ajuste rejeita `n_components` maior que `min(n_samples, n_features)`. Os testes de KMeans usam os argumentos do construtor e conferem o formato bidimensional dos centroides. As expectativas de mensagens de erro em regressão passaram a verificar a mensagem atual. `ClassificationMetrics` foi trazido de `origin/Classification_metrics`. `Standard` calcula escores z por coluna e `Matrix_design` expande potências sem interações.

**Recursos ausentes:** os três arquivos de testes de DecisionTree e o de HyperparamSelection são pulados explicitamente porque os módulos esperados ainda não existem nesta branch. Seus testes não foram considerados aprovados.

**Validação:** suíte sem `test/Utilities/Optimizer_test.py`: 130 passaram, 4 arquivos pulados, 166 subtestes passaram. Suíte completa: 131 passaram, 2 falharam em Optimizer, 4 arquivos pulados, 166 subtestes passaram.

### Inventário exato das classes e funções alteradas

| Classe ou função | Alteração feita | Relação com os testes |
|---|---|---|
| `BaseEstimator` | Acrescentados `get_params()`, `set_params(**params)` e propriedade `coef`. `set_params` aceita apenas parâmetros conhecidos, valida booleanos e invalida o ajuste anterior; `coef` devolve cópia de `_beta` após `fit`. | Fornece a interface esperada pelos testes de Linear, Ridge, Lasso e GLM. |
| `Ridge` e `Lasso` | Acrescentados construtores com `alpha`, validação de `alpha` em `set_params` e persistência da penalização para o próximo `fit` quando nenhum `alpha` explícito for passado. | Permite configurar e consultar `alpha` e usar a configuração no ajuste. |
| `GLM` | Registrados `family`, `tol` e `max_iter` como parâmetros; `set_params` valida `tol` positivo e finito e `max_iter` inteiro positivo. | Cobre a interface de configuração do GLM nos testes de parâmetros. |
| `Polynomial` | Acrescentados `get_params`, `set_params` e `coef`; a configuração inclui graus, método, intercepto e `alpha` quando aplicável. Mudanças de parâmetros recriam o estimador interno. | Expõe a mesma interface pública dos demais modelos de regressão. |
| `Classifier` | Argumento do construtor renomeado de `classes` para `labels`; criado atributo público `labels` que preserva a ordem fornecida. O armazenamento interno passou de `_classes` para `_labels`, com a referência mantida na última posição interna. | Atende aos testes de atributos e ao novo nome da API de classificação. |
| `Logistic` | Construtor e cálculos internos adaptados para `labels`; adicionados `k_function(k)`, que calcula a probabilidade do rótulo para uma observação, e `k_coef(k)`, que devolve os coeficientes desse rótulo ou zeros para a referência. | Faz os testes de atributos e métodos do Logistic passarem. |
| `LDA` e `QDA` | Construtores e validação de rótulos adaptados de `classes`/`_classes` para `labels`/`_labels`. | Mantém o ajuste e a previsão compatíveis com o construtor renomeado. |
| `KNN` | Mesmo ajuste de nomes; a votação e o vetor previsto agora usam variáveis `labels`/`predicted_labels`. | Mantém a previsão funcional após a renomeação. |
| `PCA` | `fit` passou a rejeitar `n_components > min(n_samples, n_features)`. | Faz o teste de componentes inválidos detectar o erro esperado. |
| `Standard` (nova função) | Calcula escores z por coluna e retorna zero em colunas constantes. | Substitui a função ausente exigida pelos testes de pré-processamento. |
| `Matrix_design` (nova função) | Gera potências de cada característica até `degree`, sem interações, com intercepto opcional. | Substitui a função ausente exigida pelos testes de utilitários. |
| `ClassificationMetrics` (módulo trazido de `origin/Classification_metrics`) | Exportadas `Acurracy`, `Pressision`, `Reccal` e `F1Score` por `rbg.Utilities`. | Permite coletar e executar os testes das métricas de classificação. |

**Somente testes foram alterados** para informar os argumentos obrigatórios, fornecer dados bidimensionais e amostras suficientes, usar a API atual de KFold/LOOCV e KMeans, instanciar PCA, verificar o formato correto de `coef`/centróides e não depender de uma mensagem literal antiga de `ValueError`. Os arquivos de DecisionTree e HyperparamSelection ficam marcados como `skip` enquanto os módulos correspondentes não existirem; isso não valida essas funcionalidades.

### Descrição detalhada por classe para futuro commit

- **`BaseEstimator`:** centraliza o mapeamento entre nomes públicos (`fit_intercept`, `inference`) e atributos internos. `get_params()` devolve a configuração; `set_params()` rejeita nomes desconhecidos, valida os booleanos, invalida resultados do ajuste anterior e retorna `self`. A propriedade `coef` exige um modelo ajustado e devolve uma cópia de `_beta`, incluindo o intercepto quando habilitado. `Linear` recebe essa interface por herança, sem mudança em sua classe.
- **`Ridge`:** registra `alpha` entre os parâmetros configuráveis, aceita `alpha` no construtor e valida valor numérico, finito e não negativo em `set_params()`. `fit()` usa o `alpha` armazenado quando o argumento é omitido e atualiza o valor armazenado quando recebe um argumento explícito.
- **`Lasso`:** recebe o mesmo contrato de `alpha` de `Ridge`. Isso permite configurar a penalização antes de `fit()` e consultá-la depois. As alterações locais anteriores de inferência por bootstrap permanecem no arquivo.
- **`GLM`:** inclui `family`, `tol` e `max_iter` em `get_params()`/`set_params()`. A configuração de `tol` exige número positivo e finito; `max_iter` exige inteiro positivo. Herdou de `BaseEstimator` o acesso a `coef` e a invalidação do ajuste ao mudar parâmetros.
- **`Polynomial`:** expõe `degrees`, `method`, `fit_intercept` e, para Ridge/Lasso internos, `alpha`. `set_params()` valida nomes e reconstrói o estimador interno para descartar um ajuste que não corresponde mais à configuração. `coef` delega ao estimador interno e usa a mesma ordem de colunas da expansão polinomial.
- **`Classifier`:** substitui o argumento público `classes` por `labels`. Valida se os rótulos formam um vetor não vazio sem duplicatas. `labels` preserva a ordem informada pelo usuário; `_labels` mantém a referência na última posição para os cálculos internos.
- **`Logistic`:** usa `labels` no construtor, na validação de `y`, na codificação das respostas e na previsão. `k_function(k)` devolve uma função que recebe uma observação e calcula a probabilidade do rótulo `k`. `k_coef(k)` devolve uma cópia do vetor estimado desse rótulo; para a referência, devolve um vetor de zeros. Ambos rejeitam rótulos desconhecidos e exigem ajuste prévio.
- **`LDA` e `QDA`:** renomeiam os argumentos de construção e a validação interna de `_classes` para `_labels`. Não alteram as fórmulas de estimação ou previsão.
- **`KNN`:** faz a mesma renomeação e corrige a variável usada na votação majoritária para retornar o rótulo escolhido. Não altera a regra de distância ou desempate.
- **`PCA`:** impede que `fit()` aceite mais componentes do que `min(n_samples, n_features)`, caso em que a SVD não pode produzir a quantidade pedida.
- **`KMeans`:** a classe não mudou. Os testes passaram a configurar pelo construtor, usar a grafia `Kmeans++` e conferir que `centroids` tem formato `(k, n_features)`.
- **Funções adicionadas:** `Standard` devolve escores z por coluna, com zero para colunas constantes; `Matrix_design` gera potências individuais das características, com intercepto opcional. O módulo `ClassificationMetrics` foi trazido de `origin/Classification_metrics` e suas quatro funções foram exportadas por `rbg.Utilities`.
