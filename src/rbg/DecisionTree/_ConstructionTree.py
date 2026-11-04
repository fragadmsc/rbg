import numpy as np
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode


from _RegressionTree import Regression
from _ClassificationTree import Classification


class ConstructionTree(Regression, Classification):
    """Base regression class for building a tree and fitting data."""

    def __init__(self,min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None, factor_selection = np.sqrt, random_seed = 0, form = "regression"):
        """Initializes the Regression Tree Strcuture 

                 
        Args:
        
            min_impurity_decrease: The minimum purity gain to allow a split in a node.
            min_samples_split: The minimum number of samples in a node to allow a split.
            max_depth: The maximum depth the tree can grow.
            min_samples_leaf: Minimum number of samples of each leaf. 
            max_leaf: Maximum number of leaves of the tree. 

        Initializes the attributes:
        
            - self.min_impurity_decrease: The minimum purity gain to allow a split in a node.
            - self.min_samples_split: The minimum number of samples in a node to allow a split.
            - self.max_depth: The maximum depth the tree can grow.
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
            Regression.__init__(self, criterion_split = criterion_split, custom_split =  custom_split, criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation )
        else:
            Classification.__init__(self, criterion_split = criterion_split, custom_split = custom_split , criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation =  custom_evaluation )

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
        self.factor_selection = factor_selection
        self.random_seed = random_seed

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
        self.value = np.full(self.n, None, dtype = object)
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
            seed = (self.random_seed + node) %(2**31) # to ensure that for every tree, and for every node, there is a different seed
            rng = np.random.default_rng(seed)
            predictors = int(self.factor_selection(predictors))
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
                    prevision_left = self.custom_choose(y_c_p[self.start[node]:self.start[node]+m+1]) # add +1 after m becuase in sclicing the end is exclusive
                    prevision_right = self.custom_choose(y_c_p[self.start[node]+m+1:self.end[node]]) 
                    left_mse = self.custom_split(np.array(prevision_left),np.array(y_c_p[self.start[node]:self.start[node]+m+1])) 
                    right_mse = self.custom_split(np.array(prevision_right),np.array(y_c_p[self.start[node]+m+1:self.end[node]]))
                    current_purity_gain = self.impurity[node] - ((m+1)/ (len(median) +1)* left_mse + ((len(median)+1)-(m+1))/(len(median)+1)*right_mse)



                    if current_purity_gain >= self.min_impurity_decrease: # Checkgin if the purity gain is equal or greater than the minimum necessary
                        if  gratest_purity_gain_per_p: #If a value was already computed, we have to check if the new one is better
                            if gratest_purity_gain_per_p < current_purity_gain:
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
        self.value[0] = self.custom_choose(y) # considering just regression. Might be generlized later
        self.impurity[0] = self.custom_split(self.value[0],y) # considering just regression. Might be generlized later
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
    def set_params(self, **params):
        if not params:
            pass
        else:
            defined_params = self.get_params()

            for key, value in params.items():
                if key in defined_params:
                    defined_params[key] = value
                else:
                    raise ValueError("Invalid parameter")
            self.__init__(**defined_params)

        


    def get_params(self):
        dicto = {
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_depth":self.max_depth ,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf": self.max_leaf ,
            "max_depth" :self.max_depth,
            "random_forest": self.random_forest,
            "criterion_split": self.criterion_split,
            "custom_split": self.custom_split,
            "criterion_choose": self.criterion_choose,
            "custom_choose": self.custom_choose,
            "criterion_evaluation":self.criterion_evaluation,
            "custom_evaluation": self.custom_evaluation,
            "factor_selection": self.factor_selection
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
        "aux_purity_gain":self.aux_purity_gain,
        "n": self.n
            }
        return dicto
    def _change_right_child(self,father,new_value):
        self.children_right[father] = new_value
    def _change_left_child(self,father,new_value):
        self.children_left[father] = new_value
