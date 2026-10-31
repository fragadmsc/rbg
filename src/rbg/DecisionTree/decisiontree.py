import numpy as np

from Metrics import MSE, rss, RMSE, MAE, RSquare, Gini as gini, entropy as ent, classif_error as c_e, mode
#from rgb.Utilities import MAE, RMSE, MSE, RSquare

def Gini(_, y_true):
    return gini(y_true)
def entropy(_, y_true):
    return ent(y_true)
def classif_error(_, y_true):
    return c_e(y_true)

#from rbg.Utilities import resample

from ModelSelection import resample, KFold

epsolon = 0.000000000001 # can be better defined 

class Regression:
    def __init__(self,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        split ={
            'Standard':MSE,
            'MSE':MSE,
            'RSS':rss,
            'RMSE':RMSE,
            'MAE':MAE,
            'RSquare': RSquare,
            'Other':custom_split
        }
        if criterion_split not in split:
            raise ValueError ("The only split criterions avaiable for regression are MSE, RSS, RMSE, MAE, RSquare, and Other")
        self.criterion_split  = split.get(criterion_split)

        y_pred ={
            'Standard': np.mean,
            'mean':np.mean,
            'median':np.median,
            'max':np.max,
            'min':np.min,
            'mode': mode
        }
        if criterion_choose not in split:
            raise ValueError ("The only split criterions avaiable for assinging y_pred are mean, mode, median, min, and max")
        self.criterion_choose  = y_pred.get(criterion_choose)
        evalu ={
            'MSE':MSE,
            'RSS':rss,
            'RMSE':RMSE,
            'MAE':MAE,
            'RSquare': RSquare,
            "Same": self.criterion_split,
            'Other':custom_evaluation
        } 
        if criterion_evaluation not in evalu:
            raise ValueError ("The only evalu criterions avaiable for evaluating the tree are mean, MSE, RSS, RMSE, MAE, RSquare, the Same as used in spliting and Other")
        self.criterion_evaluation = evalu.get(criterion_evaluation)
        if criterion_evaluation !="Same":
            raise Warning("It is possible, but probably inefficient to assaing a different method of evaluation")
        self.split_str = criterion_split
        self.choose_str = criterion_choose
        self.evaluation_str = criterion_evaluation
class Classification:
    def __init__(self,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        split ={
            'Standard':Gini,
            'Gini':Gini,
            'classif_error':classif_error,
            'entropy':entropy,
            'Other':custom_split
        }
        if criterion_split not in split:
            raise ValueError ("The only split criterions avaiable for classification are Gini,classif_error,entropy, and Other")
        self.criterion_split  = split.get(criterion_split)

        y_pred ={
            'Standard': mode,
            'median':np.median,
            'max':np.max,
            'min':np.min,
            'mode':mode
        }
        if criterion_choose not in split:
            raise ValueError ("The only split criterions avaiable for assinging y_pred are mode, median, min, and max")
        self.criterion_choose  = y_pred.get(criterion_choose)
        evalu ={
            'Standard':Gini,
            'Gini':Gini,
            'classif_error':classif_error,
            'entropy':entropy,
            'Same':self.criterion_split,
            'Other':custom_evaluation
        
        } 

        if criterion_evaluation not in evalu:
            raise ValueError ("The only evaluation criterions avaiable for evaluating the tree are Gini, entropy, criterion_split, the Same as used in spliting and Other")
        self.criterion_evaluation  = evalu.get(criterion_evaluation)
        if criterion_evaluation !="Same":
            raise Warning("It is possible, but probably inefficient to assaing a different method of evaluation")

        self.split_str = criterion_split
        self.choose_str = criterion_choose
        self.evaluation_str = criterion_evaluation
class ConstructionTree(Regression, Classification):
    """Base regression class for building a tree and fitting data."""

    def __init__(self,min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None, form = "regression"):
        """Initializes the Regression Tree Strcuture 

                 
        Args:
        
            min_impurity_decrease: The minimum purity gain to allow a split in a node.
            min_samples_split: The minimum number of samples in a node to allow a split.
            max_depth: The maximum depth the tree can grow.
            criterion: What type of metric the splits will be evaluated.
            min_samples_leaf: Minimum number of samples of each leaf. 
            max_leaf: Maximum number of leaves of the tree. 

        Initializes the attributes:
        
            - self.min_impurity_decrease: The minimum purity gain to allow a split in a node.
            - self.min_samples_split: The minimum number of samples in a node to allow a split.
            - self.max_depth: The maximum depth the tree can grow.
            - self.criterion: What type of metric the splits will be evaluated.
            - self.min_samples_leaf: Minimum number of samples of each leaf. 
            - self.children_left: Array mapping each node to its left child or -1 (if it is a leaf)
            - self.node_depth: Array mapping each node to its depth 
            - self.children_right: Array mapping each node to its right child or -1 (if it is a leaf)
            - self.feature: Array mapping each node to its predictor splitter 
            - self.threshold: Array mapping each node to its value of split
            - self.impurity: Array mapping each node to its impurity
            - self.value: Array mapping each node to its predictor splitter 
            - self.purity_gain_subtree: Array mapping each node to its purity gain compared to it's subtree leaves
            - self.start: Array mapping each node to the first sample that falls on that node
            - self.end: Array mapping each node to the last sample that falls on that node
            - self.aux_purity_gain: Auxiliar array mapping each node to its' purity gain compated to both children
            - self.max_leaf: Maximum number of leaves
            - self.n: Number of nodes
            - self.max_splits: Maximum number of splits
        
        Raises:

            TypeError: If the parameter 'min_impurity_decrease' is not number, or if 
            parameters 'max_depth, min_samples_split,min_samples_leaf.max_leaf 'are not integers

            ValueError: If the integer parameter if smaller than it should be to generate a tree.


        """

        if form == "regression":
            Regression.__init__(self, criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation )
        else:
            Classification.__init__(self, criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation )

        if isinstance(min_impurity_decrease, str):
             raise TypeError ("min_impurity_decrease should be a number")        
        if max_depth is not None and not isinstance(max_depth,(int,np.integer)):
            raise TypeError("The param 'ma_depth' must be None or an integer.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("'max_depth' must be greater or equal to zero.")
        if  not isinstance(min_samples_split, (int,np.integer)):
            raise TypeError('min_samples_split must be an integer.')
        if min_samples_split < 2:
            raise ValueError ('min_samples_split must be at least 2')
        if not isinstance (min_samples_leaf, (int, np.integer)):
            raise TypeError ("min_samples_leaf should be an integer")
        if min_samples_leaf <1:
            raise ValueError("min_samples_leaf must be at least one")
        if max_leaf is not None and not isinstance (max_leaf, (int, np.integer)):
            print ("olha ai o max_leaf", max_leaf)
            raise TypeError("max_leaf must be an integer")
        if max_leaf != None and max_leaf <1:
            raise ValueError("You must have at least one leaf node")    

        self.min_impurity_decrease = min_impurity_decrease 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf 
        self.children_left = None
        self.node_depth = None
        self.children_right = None
        self.feature = None 
        self.threshold = None
        self.impurity = None
        self.value = None
        self.purity_gain_subtree = None
        self.start = None
        self.end = None
        self.aux_purity_gain = None
        self.max_leaf = max_leaf
        self.n = None
        self.random_forest = random_forest
        self.nodes_bellow = None
        self.S_alg = None
        self.G_big = None
        self.g_small = None
        self.pai = None
        self.form = form
        if max_leaf:
            self.max_splits = max_leaf - 1
        else:
            self.max_splits = None
        
    def tree_structure(self,X):

        """
        Defines self.n, the number of nodes of the tree, according to the number os samples and previously
        defined limitations (min_samples_leaf, max_depth, and max_leaf). The most limited criteria will define
        how many nodes the tree will have.

        Defines the arrays size of  
            - self.children_left
            - self.children_right
            - self.feature 
            - self.node_depth 
            - self.threshold 
            - self.impurity 
            - self.value 
            - self.purity_gain 
            - self.start 
            - self.end 
            - self.values_per_leaf 
            - self.aux_purity_gain 

        Args:
            X: Sample matrix of shape (n_samples, p_feature)
        
        Returns:
            No returns. 
    
        
        """

        #conserter: tem que ser o minimo de n dado (limitação de profundidade, limitação de numero de folhas, limitação de número de amostras por folha)

        if self.min_samples_leaf > np.shape(X)[0]:
            raise ValueError ("The minimum amount of samples leaf is greater than the amount of samples. Either collect more data, or change min_sample_leaf")
        if self.max_depth:
            N = np.shape(X)[0]
            self.n = min(int(2 * np.floor(N/self.min_samples_leaf) -1),int(2 **(self.max_depth +1)) - 1 )
        else:
            N = np.shape(X)[0]
            self.n = 2 * int(np.floor(N/self.min_samples_leaf) )-1 # maybe change to ceil to be safe that it fits. Or maybe set everythng to max to guarantee.
        if self.max_leaf != None:
            self.n = min(self.n, int(2*self.max_leaf-1)) # I want the most restritive condition
        self.children_left = np.ones(self.n,dtype=int) * -1 
        self.children_right = np.ones(self.n,dtype=int) * -1 
        self.feature = np.ones(self.n,dtype=int) * -1 
        self.node_depth = np.ones(self.n,dtype=int) * -1 #impossible, so that is ok 
        self.threshold = np.full(self.n, np.nan)
        self.impurity = np.ones(self.n) * -1 
        self.value = np.full(self.n, np.nan)
        self.purity_gain = np.ones(self.n) * -1 
        self.start = np.ones(self.n,dtype=int) * -1
        self.end = np.ones(self.n,dtype=int) * -1
        self.values_per_leaf = np.arange(self.n)# tirar
        self.aux_purity_gain = np.ones(self.n,dtype=int) * -1 
        self.nodes_bellow = np.ones(self.n,dtype=int) * -1 
        self.S_alg= np.ones(self.n,dtype=int) * -1
        self.G_big = np.full(self.n, None, dtype=object)
        self.g_small = np.ones(self.n,dtype=float) * -1.0
        self.pai = np.ones(self.n,dtype=int) * -1 
        if self.max_leaf == None:
            self.max_leaf = (self.n+ 1)//2  # talvez tenha um +1 ou -1 ai, sei la. Mas eh o maximo de folhas que poderiamos ter. Como eu tenho o minimo de nos por folha...
            self.max_splits = self.max_leaf - 1

    def splitter(self, X,y,node,defenitive = False):

        """

        Algorithm designed to find the best predictor and the best value of that predictor
        to partitionate a node. 
        If the node is found to be the best one to be splitted, it will indeed be partitioned and 
        generate two new nodes.

        For a fixed node, the algorithm will iterate for every parameter. In each parameter, 
        the method "_best_cut_value_for_fixed_p" will be called to find the best cut value 
        of that parameter ( if it existis). Then, the chosen parameter to cut the node will be
        selected. If defenitive = False, only the information of the impurity decrease will be returned.
        If defenitive = True, it means that the node will actually be cut, therefore, its children
        will be defined, and the node's attributes of threshold and features will be updated. Also, 
        the sample matrix "X" will be updated, that is, its lines will re-arrenge themselves to simulate
        the final partition of the sample datas in each leaf. Therefore, the y vector will have its
        lines equally re-arrenged. Lsatly, it sets up left child and right child as possible future nodes
        to be split.


        Args:
            X: Sample matrix of shape (n_samples, p_feature).
            y: Vector of the observations of shape (n_samples,).
            node: indice of the node that will be testedo or actually partitioned
            defenitive: if "False" then splitter only find the best cut of the node. If "True", then
            besides finding the best, it will actually cut the node and generate the new leaves.
        
        Returns:
            None - > If there is no possible cut 
            gratest_purity_gain_overall -> If there is a possible cut, returns the onw which
            increses the purity the most


        
        """

        matri_list = [] 
        possible_cuts = [] 
        y_list = [] 

        y_tam = int(y.shape[0])

        predictors = int(X.shape[1])

        train_idxs = np.arange(predictors)


        if self.random_forest:
            all_idxs = np.arange(predictors)
            rng = np.random.default_rng(node)
            predictors = int(np.sqrt(predictors))
            train_idxs = rng.choice(all_idxs, size=predictors, replace=False)




        for predictor in train_idxs:

            X_c_p = X.copy()
            lines = X[self.start[node]:self.end[node]] #lines from the sample that are on the current node
            ordered_ind = np.argsort(lines[:,predictor]) # incrising order index of those lines according to the predictor
            final_lines = lines[ordered_ind] # re - arranged lines that are on "node" but in increasing order according to precidctor
            col_pred = final_lines[:, predictor]

            X_c_p [self.start[node]:self.end[node]] = final_lines # re - arrenged matrix X

            y_c_p = y.copy()
            y_lines = y[self.start[node]:self.end[node]]
            final_y_lines = y_lines[ordered_ind]
            y_c_p [self.start[node]:self.end[node]] = final_y_lines

            median = (col_pred[:-1] + col_pred[1:])/2 # Candidated for cut value. This will have to be chenged for classification (probably)

            self._best_cut_value_for_fixed_p(node,median,matri_list,possible_cuts,y_list,X_c_p,y_c_p,predictor,final_lines)


        y_list = np.array(y_list)
        matri_list = np.array(matri_list)
        possible_cuts = np.array(possible_cuts)


        if y_list.shape[0] == 0:
            return None 
        else:
            gratest_purity_gain_overall_index = np.argmax(possible_cuts[:,0])
            gratest_purity_gain_overall = possible_cuts[gratest_purity_gain_overall_index,0]
            if defenitive == False:
                return gratest_purity_gain_overall
            else:
                X = matri_list[gratest_purity_gain_overall_index]
                y = y_list[gratest_purity_gain_overall_index]
                self.threshold[node] = possible_cuts[gratest_purity_gain_overall_index,1]
                self.feature[node] = possible_cuts[gratest_purity_gain_overall_index,2]
                r1 = np.max(self.children_right)
                r2 = np.max(self.children_left)
                right = max(r1,r2)+1
                left = right+1
                if right == 0: #because in the first split, all values will be -1, but 0, 0 = -1 +1, is the root.
                    right =1 
                    left = 2         

                self.children_left[node] = left
                self.children_right[node] = right

                self.value[left] = possible_cuts[gratest_purity_gain_overall_index,6]
                self.impurity[left] = possible_cuts[gratest_purity_gain_overall_index,4]
                self.start[left] = self.start[node]
                self.end[left] = possible_cuts[gratest_purity_gain_overall_index,3]
                self.node_depth[left] = self.node_depth[node] +1 
                self.pai[left] = node

                if (self.end[left] - self.start[left]) >= self.min_samples_split and (self.max_depth == None or self.node_depth[left] < self.max_depth) and self.impurity[left] > epsolon:
                    self.aux_purity_gain[left] = 1

                self.value[right] = possible_cuts[gratest_purity_gain_overall_index,7]
                self.impurity[right] = possible_cuts[gratest_purity_gain_overall_index,5]
                self.start[right]= possible_cuts[gratest_purity_gain_overall_index,3] #might need to review that
                self.end[right] = self.end[node]
                self.node_depth[right] = self.node_depth[node] +1 
                self.pai[right] = node

                if (self.end[right] - self.start[right]) >= self.min_samples_split and (self.max_depth == None or self.node_depth[right] < self.max_depth) and self.impurity[right] > epsolon:
                    self.aux_purity_gain[right] = 1
                return gratest_purity_gain_overall   


    def _best_cut_value_for_fixed_p(self,node,median,matri_list,possible_cuts,y_list,X_c_p,y_c_p,predictor,final_lines):
            """

            For a given predictior and node, the module finds that best cut value ( if it exists), and then:
                - add to 'matri_list' the new X line ordered formed after the possible split (X_c_p)
                - add to 'y_list' the new y line ordered (y_c_p) to match X_c_p
                - add to 'possible_cuts' the best cut found, which is a list containing:
                    * gratest_purity_gain_per_p: The best burity gain, over all value cuts, of the fixed parameter
                    * associated_value_per_p: The value cut that maximized the purity gain
                    * predictor: The fixed predictor
                    * new_end_begin: The samples that originally belonged to the node ( that is being checked if can be splitted)
                    will be divided between the left child and right child ( if the node is splitted). Since in X_c_p those samples
                    are ordered according to the fixed parameter, then "new_end_begin" delimits the new "slice" of samples that
                    belongs to the left child and right child. 
                    * imp_left: impurity of the left child, according to this parameter
                    * imp_right: impurity of the right child, according to this parameter
                    * prev_l: prediction of the left child, according to this parameter
                    * prev_r: prediction of the right child, according to this parameter
            
            If, for a given cut value, the split violates any of the conditions 
                -'The minimum purity gain to allow a split in a node.'
                -'Minimum number of samples of each leaf.'
            Then this value cut will not be further considered to actually partition the sample space. 
            If all of the cut values are invalid, then 'gratest_purity_gain_per_p' will remain as None,
            and the fixed parameter will not be considered to partitionate the node. 

            The current metrics for prediction is the mean values of the samples that fall on the region,
            and the impurity is measured with MSE (so far).

            Since the sequence of the possible cut values is calculated based on the ordinated sequence of 
            the fixed predictor, there are (self.end[node] - self.start[node] -1) possible values. 
            For Example, if "pi" is the predictor value of the samples, ordinated in final_lines , and "mj" the median 
            values calculated as mj = (pj + p(j+1))/2, we would have:

                    p1  p2   p3  p4  p5  p6  p7  p8  p9  p10 
                      m1  m2   m3  m4  m5  m6  m7  m8  m9

            Some "mj" values might be repeated. For example, if p1 = p2 = p3, then m1 = m2. Therefore,
            in this cenario, instead of dividing de sample values according to the position m1, that is,
            instead of testing 

                    p1 | p2   p3  p4  p5  p6  p7  p8  p9  p10 
                      m1   m2   m3  m4  m5  m6  m7  m8  m9
            Since m1 = m2, and values are ordered, it means that the samples that would be on the left
            child according to m1, should be actually the same according to m2. Therefore, whenever we are 
            testing those "mj" values, we will take the last indice "k" from "median" , such that mj = mk,
            and we will compare mk to p(k+1). If mk = p(k+1), then the cut on the sample will be indice 
            "k+1":

                    p1  p2  p3 | p4  p5  p6  p7  p8  p9  p10 
                      m1  m2  m3  m4  m5  m6  m7  m8  m9

            Args:
                - node: node evaluated for the split
                - median: sequence of the possible cut values. 
                - matri_list: list of the X with its lines re-ordered to match the split
                - possible_cuts: list of the possible cuts of the node
                - y_list: list of the y with its lines re-ordered to match the split
                - X_c_p: Matrix X with the lines re-ordered to match the split according to a fixed predictor
                - y_c_p: vector y with the lines re-ordered to match the split according to a fixed predictor
                - predictor: the fixed predictor
            

            
            """
            m = 0
            gratest_purity_gain_per_p = None
            associated_value_per_p = None
            new_end_begin = None
            imp_left = None
            imp_right = None
            prev_l= None
            prev_r = None
            while m < len(median):
                id_last_m = np.searchsorted(median, median[m], side = 'right') - 1  #since we are using an ordinated array, we can use a function that uses binary search to avoid recounting repeated medians.
                if median[m] == median[id_last_m]:
                    m = id_last_m 
                if median[m] == final_lines[m+1,predictor]:
                    if m+1 == len(median): #Extremly important break. If it is true, then it means that we would try to split by letting 
                        break # everything on the left child. Besides being prohibited, it would cause index error in the next lines.
                    else:
                        m+=1
                #prevision_left = np.mean(y_c_p[self.start[node]:self.start[node]+m+1]) # add +1 after m becuase in sclicing the end is exclusive
                #prevision_right = np.mean(y_c_p[self.start[node]+m+1:self.end[node]]) 
                
                if m+1 >=  self.min_samples_leaf and len(median)+1 - (m +1) >= self.min_samples_leaf: # checking if the left child and right child both have the minimum amount of samples
                    prevision_left = self.criterion_choose(y_c_p[self.start[node]:self.start[node]+m+1]) # add +1 after m becuase in sclicing the end is exclusive
                    prevision_right = self.criterion_choose(y_c_p[self.start[node]+m+1:self.end[node]]) 
                    left_mse = self.criterion_split(np.array(prevision_left),np.array(y_c_p[self.start[node]:self.start[node]+m+1])) 
                    right_mse = self.criterion_split(np.array(prevision_right),np.array(y_c_p[self.start[node]+m+1:self.end[node]]))
                    current_purity_gain = self.impurity[node] - ((m+1)/ (len(median) +1)* left_mse + ((len(median)+1)-(m+1))/(len(median)+1)*right_mse)
                    if current_purity_gain >= self.min_impurity_decrease: # Checkgin if the purity gain is equal or greater than the minimum necessary
                        if  gratest_purity_gain_per_p: #If a value was already computed, we have to check if the new one is better
                            if gratest_purity_gain_per_p >current_purity_gain:
                                gratest_purity_gain_per_p = current_purity_gain
                                associated_value_per_p = median[m]
                                new_end_begin = self.start[node] +m+1 # considerind [start , end)
                                imp_left = left_mse
                                imp_right = right_mse
                                prev_l= prevision_left
                                prev_r = prevision_right
                        else: # If no value was computed, the current is the best
                            gratest_purity_gain_per_p = current_purity_gain
                            associated_value_per_p = median[m]
                            new_end_begin = self.start[node] +m+1 # considerind [start , end)
                            imp_left = left_mse
                            imp_right =right_mse
                            prev_l= prevision_left
                            prev_r = prevision_right
                m +=1
            if gratest_purity_gain_per_p:
                matri_list.append(X_c_p)
                possible_cuts.append([gratest_purity_gain_per_p,associated_value_per_p,predictor,new_end_begin,imp_left, imp_right,prev_l,prev_r])
                y_list.append(y_c_p)    

    def tree_construction(self, X,y):

        """

        Builds a decision tree from the matrix X and vector y. 
        It calls the tree_structure to build the tree strcuture. Then, initializes the 
        root node (0). Then, the algorithm checks if it is avaiable to cut the root
        according to the limitations of the tree. It uses the function "splitter" to findo
        the best split of the root, if it is applicable. Then, if "splitter" found a good split,
        and if there is still avaiable splits, the algorithm will perform the split. This process
        will continue in loop until there is no more splits avaiable according to the limitatinos
        that defined the tree and while "splitter" still finds avaiable partitions.Since there are
        parameters such as " maximum number of leaves", there could have, therefore, a maximum number
        of splits. Hence, the splits that will actually be performed, are the ones from the nodes that
        at the given step of the algorith, provide the greatest purity gain. 
        

        Args:
            X: Matrix of shape (n_new_samples, p_features)
            y: Vector of the observations of shape (n_samples,).
        
        Raises:
            ValueError: if the number of dimensions os X and y are incompatible

        
        """
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match")
        self.tree_structure(X)
        self.value[0] = self.criterion_choose(y) # considering just regression. Might be generlized later
        self.impurity[0] = self.criterion_split(self.value[0],y) # considering just regression. Might be generlized later
        self.start[0] = 0
        self.end[0] = X.shape[0] # considerind [start , end)
        self.node_depth[0] = 0

        list_nodes_that_can_be_split =  [] 

        if (self.max_leaf >1 or self.max_leaf == None) and  X.shape[0] >=  self.min_samples_split and (self.max_depth == None or self.max_depth >=1 ) and self.impurity[0]>epsolon:
            gain = self.splitter(X,y,0)
            if gain:
                list_nodes_that_can_be_split.append([gain,0])
                self.aux_purity_gain[0] = 1

    
        while ( 1 in self.aux_purity_gain and self.max_splits>0):
            if len(list_nodes_that_can_be_split) >0:
              
                chosen = max(list_nodes_that_can_be_split, key = lambda x:x[0]) 
                node_chosen_to_split = chosen[1]
                list_nodes_that_can_be_split.remove(chosen)
              
                self.aux_purity_gain[node_chosen_to_split] = -1 # turning it back to -1 
                u = self.splitter(X,y,node_chosen_to_split, defenitive= True)
                self.max_splits -=1

                if self.aux_purity_gain[self.children_left[node_chosen_to_split]] == 1: 
                    new_candidate = self.splitter(X,y,self.children_left[node_chosen_to_split])
                    if new_candidate: # That is, if it returned a value different from "None" and gain > 0
                        list_nodes_that_can_be_split.append([new_candidate,self.children_left[node_chosen_to_split]])

                    else: # if in reality, we can not split it
                        self.aux_purity_gain[self.children_left[node_chosen_to_split]] = -1 
                if self.aux_purity_gain[self.children_right[node_chosen_to_split]] == 1: 
                    new_candidate = self.splitter(X,y,self.children_right[node_chosen_to_split])
                    if new_candidate:# That is, if it returned a value different from "None" and gain > 0
                        list_nodes_that_can_be_split.append([new_candidate,self.children_right[node_chosen_to_split]])

                    else:
                        self.aux_purity_gain[self.children_right[node_chosen_to_split]] = -1

    def predict (self,X):

        """
        After the tree has been built, X will be predicted line by line, by "following the tree".

        Args:
            X: Matrix
        
        Returns:
            y: np.ndarray of the predicted values
        
        """

        if X.ndim == 1:
            X= X.reshape(1,-1)

        y_num = X.shape[0]
        
        y = [] 
        for i in range(y_num):
            node = 0
            while self.children_left[node] != -1: # Because if it has left child, then it has right child
                if X[i][self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            pred_y = self.value[node]
            y.append(pred_y)
        y = np.array(y)
        return y
    def set_params(self, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None):

        if isinstance(min_impurity_decrease, str):
             raise TypeError ("min_impurity_decrease should be a number")        
        if max_depth is not None and not isinstance(max_depth,(int,np.integer)):
            raise TypeError("The param 'ma_depth' must be None or an integer.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("'max_depth' must be greater or equal to zero.")
        if  not isinstance(min_samples_split, (int,np.integer)):
            raise TypeError('min_samples_split must be an integer.')
        if min_samples_split < 2:
            raise ValueError ('min_samples_split must be at least 2')
        if not isinstance (min_samples_leaf, (int, np.integer)):
            raise TypeError ("min_samples_leaf should be an integer")
        if min_samples_leaf <1:
            raise ValueError("min_samples_leaf must be at least one")
        if max_leaf is not None and not isinstance (max_leaf, (int, np.integer)):
            raise TypeError("max_leaf must be an integer")
        if max_leaf != None and max_leaf <1:
            raise ValueError("You must have at least one leaf node")    

        self.min_impurity_decrease = min_impurity_decrease 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf 
        self.children_left = None
        self.node_depth = None
        self.children_right = None
        self.feature = None 
        self.threshold = None
        self.impurity = None
        self.value = None
        self.purity_gain_subtree = None
        self.start = None
        self.end = None
        self.aux_purity_gain = None
        self.max_leaf = max_leaf
        self.n = None
        if max_leaf:
            self.max_splits = max_leaf - 1
        else:
            self.max_splits = None
    def get_params(self):
        dicto = {
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_depth":self.max_depth ,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf": self.max_leaf ,
            "max_splits": self.max_splits
            }
        return dicto
    def get_tree_structure_info(self):
        dicto = {
        "children_left":self.children_left,
        "children_right":self.children_right,
        "feature":self.feature,
        "node_depth":self.node_depth ,
        "threshold":self.threshold,
        "impurity":self.impurity,
        "value":self.value,
        "purity_gain":self.purity_gain,
        "start":self.start,
        "end":self.end,
        "aux_purity_gain":self.aux_purity_gain
            }
        return dicto
    def _change_right_child(self,father,new_value):
        self.children_right[father] = new_value
    def _change_left_child(self,father,new_value):
        self.children_left[father] = new_value
    

                         

class TreeBase(ConstructionTree):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None,  form = "regression" ): 
        super().__init__( min_impurity_decrease , min_samples_split , max_depth ,criterion , min_samples_leaf , max_leaf , random_forest, criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation,form )
    def fit(self,X,Y, prune =True):
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
            tree = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth,criterion = self.criterion, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf, random_forest = False,criterion_split = self.split_str , custom_split = self.criterion_split, criterion_choose = self.choose_str , criterion_evaluation = self.evaluation_str , custom_evaluation = self.criterion_evaluation,  form = self.form )#tem que adicionar aqui os mesmos parametros 
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
            mean_of_mse_of_y_hat = self.criterion_choose([self.criterion_evaluation(np.array(trees[i].predict(list_test_X[i])),np.array(list_test_y[i])) for i in range(len(trees))])
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
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None, form = "regression"):
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
        super().__init__(min_impurity_decrease , min_samples_split , max_depth ,criterion , min_samples_leaf , max_leaf, random_forest,criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation,form)
    def fit(self, X,y, random_forest = False):

        r_f = random_forest

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

        for _ ,(train, test) in enumerate(resamples):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            tree_test = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth,criterion = self.criterion, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf,random_forest=r_f,criterion_split = self.split_str , custom_split = self.criterion_split, criterion_choose = self.choose_str , criterion_evaluation = self.evaluation_str , custom_evaluation = self.criterion_evaluation,  form = self.form)
            tree_test.fit(X_train,y_train,prune=False)
            self.list_of_trees.append(tree_test)
            self.X_trees_train.append(X_train)
            self.y_trees_train.append(y_train)
            self.X_trees_test.append(X_test)
            self.y_trees_test.append(y_test)
            self.list_of_indx_test.append(test)
    def predict(self, X_test, trees = None):
        if trees == None:
            trees = self.list_of_trees
        y_prediction = self.criterion_choose([tree.predict(X_test) for tree in trees])
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
        return self.criterion_evaluation( y_hat_array,np.array(self.y_orig))

class RandomForestBase(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None,form = "regression"):
        super().__init__(n_trees, min_impurity_decrease , min_samples_split , max_depth ,criterion , min_samples_leaf, max_leaf , random_forest ,criterion_split, custom_split, criterion_choose , criterion_evaluation , custom_evaluation,form)
    def fit(self, X,y):
        super().fit(X,y, random_forest = True)

class RegressionTree(TreeBase):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None ):
        super().__init__( min_impurity_decrease , min_samples_split , max_depth ,criterion , min_samples_leaf , max_leaf , random_forest, criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation,form = "regression")


class ClassificationTree(TreeBase):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None ):
        super().__init__( min_impurity_decrease , min_samples_split , max_depth ,criterion , min_samples_leaf , max_leaf , random_forest, criterion_split , custom_split , criterion_choose , criterion_evaluation , custom_evaluation,form = "classification")

class BaggingReg(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        super().__init__(n_trees, min_impurity_decrease, min_samples_split, max_depth ,criterion, min_samples_leaf , max_leaf, random_forest,criterion_split, custom_split, criterion_choose , criterion_evaluation , custom_evaluation , form = "regression")
class Bagging(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        super().__init__(n_trees, min_impurity_decrease, min_samples_split, max_depth ,criterion, min_samples_leaf , max_leaf, random_forest,criterion_split, custom_split, criterion_choose , criterion_evaluation , custom_evaluation , form = "classification")


class RandomForestRegression(RandomForestBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        super().__init__(n_trees, min_impurity_decrease, min_samples_split, max_depth ,criterion, min_samples_leaf , max_leaf, random_forest,criterion_split, custom_split, criterion_choose , criterion_evaluation , custom_evaluation , form = "regression")
class RandomForest(RandomForestBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", criterion_evaluation = "Same", custom_evaluation = None):
        super().__init__(n_trees, min_impurity_decrease, min_samples_split, max_depth ,criterion, min_samples_leaf , max_leaf, random_forest,criterion_split, custom_split, criterion_choose , criterion_evaluation , custom_evaluation , form = "classification")
