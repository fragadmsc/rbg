import numpy as np
from rbg.Utilities import MSE, rss, RMSE, MAE, RSquare
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode
from ModelSelection import resample, KFold

from _ConstructionTree import ConstructionTree

def Gini(_, y_true):
    return gini(y_true)
def entropy(_, y_true):
    return ent(y_true)
def classif_error(_, y_true):
    return c_e(y_true)


epsolon = 0.000000000001 
    

                         

class TreeBase(ConstructionTree):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt , random_seed = 0, form = "regression" ): 
        super().__init__( min_impurity_decrease =min_impurity_decrease , min_samples_split = min_samples_split , max_depth = max_depth  , min_samples_leaf =min_samples_leaf , max_leaf =  max_leaf , random_forest = random_forest, criterion_split = criterion_split , custom_split = custom_split , criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation, factor_selection = factor_selection,random_seed =random_seed, form=form  )
    def fit(self,X,Y, prune =False):
        super().tree_construction(X,Y)
        if prune:
            self.prune(X,Y)
    def prune(self,X,Y):
        """

        The methor of prune follows the Minimal Cost-Complexity Pruning. 

        First, it will take the list of possible "penalizator" values and its associated trees from
        the original main tree calculated in fit. But, the tree will not be cut. Instead, in the same
        list the "penalizators" will be saved in increasing order, _alfas,  the note that will become a leaf will
        be joined, as in a tuple. 

        Then, it performs a K-fold on X. After performing the k-fold on X, we will do the same 
        process as we made for the entire X, but on each of those trees. They will also not be cut straighfoward.

        Then, for each alfa in _alfas, and for each k-fold tree, we will search in which interval
        alfa falls in. When we find where it falls, we simply take children_left[node] = -1 and children_right[node] = -1
        for every node in the tuple (alfa, node) in _alfa which is smaller or equal than the one from our interval. 

        Then, after cuting those k-fols, we will test each of them with their k-fold. Then, perform
        MSE in the results individually. Then, it takes the median of those MSE, and saves the median,
        in the error_list. 

        At the end, the indx with the smallest index, will be the chosen alfa ( and its associated tree). 
        Now that the algorith knows where to cut the original tree, it will actually cut the tree, by 
        making the children_left[node] = -1 and children_right[node] = -1 for every node in 
        (alfa, node) with alfa <= chosen_alfa. 

        Args:
            X
            y
        
        """

        alfas_oficial = self.cuts_and_alfas()

        number_splits = min(5,Y.shape[0])
        k_fold = KFold(random_state = 41, n_splits = number_splits)
        k = k_fold.split(X.copy())

        list_test_X =[] 
        list_train_X =[] 
        list_test_y =[] 
        list_train_y =[] 

        list_errors = [] 

        for _, (train, test) in enumerate(k):
            X_train, X_test = X[train], X[test]
            y_train, y_test = Y[train], Y[test]
            list_test_X.append(X_test)
            list_train_X.append(X_train)
            list_test_y.append(y_test)
            list_train_y.append(y_train) 

        trees = [] 
        alfas_list = [] 

        for t in range(len(list_test_X)): 
            tree = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf, random_forest = False,criterion_split = self.criterion_split , custom_split = self.custom_split, criterion_choose = self.criterion_choose , criterion_evaluation = self.criterion_evaluation , custom_evaluation = self.custom_evaluation, factor_selection = self.factor_selection,  form = self.form )#We have to give the same params 
            tree.fit(list_train_X[t], list_train_y[t],prune = False)
            alfa_list_fold = tree.cuts_and_alfas()
            alfas_list.append(alfa_list_fold)
            trees.append(tree)

        for [alfa,node] in alfas_oficial:
            for i in range(len(trees)):
                for ind in range(len(alfas_list[i])):
                    if alfas_list[i][ind][0]<= alfa:
                        if alfas_list[i][ind][1] != None:
                            trees[i]._change_right_child(alfas_list[i][ind][1],-1)
                            trees[i]._change_left_child(alfas_list[i][ind][1],-1)
                    else:
                        break # Inspect another tree
            #After prunning the trees, we will predict the y_test and then take the MSE
            mean_of_mse_of_y_hat = self.custom_choose([self.custom_evaluation(np.array(trees[i].predict(list_test_X[i])),np.array(list_test_y[i])) for i in range(len(trees))])
            list_errors.append(mean_of_mse_of_y_hat)
        list_errors = np.array(list_errors)
        alfa_index = np.argmin(list_errors)
        #defenitive prunning
        for ind in range(len(alfas_oficial)):
            if ind <= alfa_index:
                [alfa,node] = alfas_oficial[ind]
                if node != None:
                    self.children_right[node] = -1 
                    self.children_left[node] = -1 
            else:
                break 
        
    
    def cuts_and_alfas(self):

        def find_N(node):
            if self.children_left[node] != -1:
                self.nodes_bellow[node] = find_N(self.children_left[node])+find_N(self.children_right[node])
            else:
                self.nodes_bellow[node] = 1
            return self.nodes_bellow[node]
        def find_S (node):
            if self.children_left[node] != -1:
                self.S_alg[node] = find_S(self.children_left[node])+find_S(self.children_right[node])
            else:
                self.S_alg[node] = (self.impurity[node])*(self.end[node] - self.start[node])/(self.end[0])
            return self.S_alg[node]
        def find_g(node):
            if self.children_left[node] != -1:
                self.g_small[node] = ((self.impurity[node]*(self.end[node] - self.start[node])/(self.end[0])) - self.S_alg[node])/(self.nodes_bellow[node] - 1)
                find_g(self.children_left[node])
                find_g(self.children_right[node])
            else:
                self.g_small[node] = np.inf
            return self.g_small[node]

        def find_G(node):
            if self.children_left[node] != -1:

                L = find_G(self.children_left[node])
                R = find_G(self.children_right[node])

                if R[0] > L[0]:
                    self.G_big[node]=[L[0],L[1]] 
                else:
                    self.G_big[node] = [R[0],R[1]]

                if self.G_big[node][0] >self.g_small[node]:
                    self.G_big[node] = [self.g_small[node],node]


            else:
                self.G_big[node] = [np.inf,node]
            return self.G_big[node]

        _ = find_N(0)   
        _ = find_S(0)
        _ = find_g(0)
        _ = find_G(0)

        alfa_list = [[0,None]]

        while self.nodes_bellow[0] >1:
            if self.G_big[0][0] > max(alfa_list, key = lambda x:x[0])[0]+epsolon:
                alfa_list.append(self.G_big[0])
            copi = self.nodes_bellow[self.G_big[0][1]] 
            self.nodes_bellow[self.G_big[0][1]] = 1 
            self.S_alg[self.G_big[0][1]] = (self.impurity[self.G_big[0][1]])*(self.end[self.G_big[0][1]] - self.start[self.G_big[0][1]])/(self.end[0]) 
            self.G_big[self.G_big[0][1]] = [np.inf,self.G_big[0][1]]
            pr_node = self.G_big[0][1]

            while pr_node != 0:
                pr_node= self.pai[pr_node]
                self.nodes_bellow[pr_node] =  self.nodes_bellow[self.children_left[pr_node]]+self.nodes_bellow[self.children_right[pr_node]]
                self.S_alg[pr_node] = self.S_alg[self.children_left[pr_node]]+self.S_alg[self.children_right[pr_node]]
                self.g_small[pr_node] = ((self.impurity[pr_node]*(self.end[pr_node] - self.start[pr_node])/(self.end[0])) - self.S_alg[pr_node])/(self.nodes_bellow[pr_node] - 1)
                L = self.G_big[self.children_left[pr_node]]
                R = self.G_big[self.children_right[pr_node]]
                
                if R[0] > L[0]:
                    self.G_big[pr_node]=[L[0],L[1]] 
                else:
                    self.G_big[pr_node] = [R[0],R[1]]
                
                if self.G_big[pr_node][0] >self.g_small[pr_node]:
                    self.G_big[pr_node] = [self.g_small[pr_node],pr_node]               


         

        return alfa_list




          

class BaggingBase(TreeBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None,  factor_selection = np.sqrt, form = "regression"):
        self.n_trees = n_trees
        if not isinstance(n_trees,int):
            raise TypeError("number of trees must be integer")
        if n_trees < 1:
            raise ValueError("number of trees must be at least one")
        self.X_trees_train =[]
        self.y_trees_train = [] 
        self.X_trees_test =[]
        self.y_trees_test = [] 
        self.list_of_trees = []
        self.list_of_indx_test = [] 
        self.qnt_samples = 0
        self.X_orig =None
        self.y_orig = None
        super().__init__(min_impurity_decrease = min_impurity_decrease , min_samples_split = min_samples_split  , max_depth = max_depth , min_samples_leaf = min_samples_leaf , max_leaf = max_leaf, random_forest = random_forest,criterion_split=criterion_split , custom_split=custom_split , criterion_choose=criterion_choose , custom_choose=custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation,factor_selection = factor_selection, form=form )
    def fit(self, X,y):
        X = np.asarray(X)
        y = np.asarray(y)


        self.X_orig =X
        self.y_orig = y
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match")

        self.qnt_samples = X.shape[0]
        resamples = resample(X,n_resamples = self.n_trees,random_state = 42)
        number_of_the_tree = 0 

        for _ ,(train, test) in enumerate(resamples):
            number_of_the_tree+=1
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            tree_test = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf,random_forest=self.random_forest,criterion_split = self.criterion_split , custom_split = self.custom_split, criterion_choose = self.criterion_choose , criterion_evaluation = self.criterion_evaluation , custom_evaluation = self.custom_evaluation,  factor_selection = self.factor_selection, random_seed = number_of_the_tree, form = self.form)
            tree_test.fit(X_train,y_train,prune=False)
            self.list_of_trees.append(tree_test)
            self.X_trees_train.append(X_train)
            self.y_trees_train.append(y_train)
            self.X_trees_test.append(X_test)
            self.y_trees_test.append(y_test)
            self.list_of_indx_test.append(test)
    def predict(self, X_test, trees = None):
        X_test = np.asarray(X_test)
        if trees == None:
            trees = self.list_of_trees
        y_preds = [tree.predict(X_test) for tree in trees]
        y_prediction = self.custom_choose(y_preds,axis = 0)
        return np.array(y_prediction)
    def OOB(self):
        y_hat_array = [] 
        for n in range(self.qnt_samples):
            tree_list = [] 
            for l in range(len(self.list_of_indx_test)):
                if n in self.list_of_indx_test[l]:
                    tree_list.append(self.list_of_trees[l])
            y_hat = self.predict(self.X_orig[n], trees = tree_list)
            y_hat_array.append(y_hat)
        y_hat_array = np.array(y_hat_array)
        return self.custom_evaluation( y_hat_array,np.array(self.y_orig))

class RandomForestBase(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None,factor_selection = np.sqrt,form = "regression"):
        super().__init__(n_trees = n_trees, min_impurity_decrease = min_impurity_decrease, min_samples_split = min_samples_split , max_depth = max_depth , min_samples_leaf = min_samples_leaf, max_leaf = max_leaf , random_forest = random_forest ,criterion_split = criterion_split, custom_split = custom_split, criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation,factor_selection = factor_selection, form = form)






class RegressionTree(TreeBase):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None , factor_selection = np.sqrt):
        super().__init__( min_impurity_decrease = min_impurity_decrease , min_samples_split = min_samples_split, max_depth = max_depth  , min_samples_leaf = min_samples_leaf , max_leaf =  max_leaf , random_forest =  random_forest, criterion_split = criterion_split, custom_split=custom_split , criterion_choose=criterion_choose , custom_choose=custom_choose , criterion_evaluation = criterion_evaluation, custom_evaluation= custom_evaluation,factor_selection=factor_selection, form = "regression")

class ClassificationTree(TreeBase):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt ):
        super().__init__( min_impurity_decrease =  min_impurity_decrease  , min_samples_split =min_samples_split , max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf= max_leaf , random_forest=random_forest, criterion_split=criterion_split , custom_split=custom_split , criterion_choose =criterion_choose , custom_choose=custom_choose, criterion_evaluation=criterion_evaluation , custom_evaluation=custom_evaluation, factor_selection=factor_selection, form = "classification" )

class BaggingReg(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf= max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split=custom_split, criterion_choose=criterion_choose , custom_choose=custom_choose, criterion_evaluation =criterion_evaluation , custom_evaluation =custom_evaluation, factor_selection=factor_selection, form = "regression")
    def get_params(self):
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
class Bagging(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf=max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split=custom_split, criterion_choose=criterion_choose , custom_choose= custom_choose, criterion_evaluation=criterion_evaluation , custom_evaluation=custom_evaluation , factor_selection=factor_selection, form = "classification")
    def get_params(self):
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
    


class RandomForestRegression(RandomForestBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf =min_samples_leaf , max_leaf=max_leaf, random_forest =random_forest,criterion_split=criterion_split, custom_split =custom_split, criterion_choose =criterion_choose , custom_choose = custom_choose, criterion_evaluation =criterion_evaluation , custom_evaluation =custom_evaluation , factor_selection = factor_selection, form = "regression")
    def get_params(self):
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
class RandomForest(RandomForestBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees = n_trees, min_impurity_decrease = min_impurity_decrease, min_samples_split = min_samples_split, max_depth=max_depth , min_samples_leaf = min_samples_leaf , max_leaf=max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split= custom_split, criterion_choose=criterion_choose ,custom_choose=custom_choose, criterion_evaluation =criterion_evaluation, custom_evaluation=custom_evaluation , factor_selection=factor_selection, form = "classification")
    def get_params(self):
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
