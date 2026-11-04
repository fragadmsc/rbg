import numpy as np

def Gini(y_true:np.ndarray) :

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    ind_gin = sum(pmk*(1-pmk) for pmk in p)

    return ind_gin


def classif_error(y_true:np.ndarray):

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    return (1 -np.max(p))

    


def entropy(y_true:np.ndarray) :

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    entropy_result = sum(pmk*np.log(pmk) for pmk in p)


    return - entropy_result

 

def mode(y_true:np.ndarray, axis = None):

    y_true = np.asarray(y_true)

    if axis is not None and y_true.ndim == 2:
        n_samples = y_true.shape[1]
        mode_results = []
        for i in range(n_samples):
            previsions_or_i = y_true[:,i]
            values, cont = np.unique(previsions_or_i, return_counts=True)
            ind = np.argmax(cont)
            mode_results.append(values[ind])
        return np.array(mode_results)

    else:
        values, cont = np.unique(y_true, return_counts = True)
        ind = np.argmax(cont)
        return values[ind]

