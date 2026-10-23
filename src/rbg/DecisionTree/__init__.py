import numpy as np

from Metrics import MSE


epsolon = 0.00000001 # can be better defined 


class Regression:
    """Base regression class for building a tree and fitting data. """

    def __init__(self,min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None,criterion = 'SQE', min_samples_leaf = 1, max_leaf = None):
        "Initialized tree "
        self.min_impurity_decrease = min_impurity_decrease 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = max(min_samples_leaf,1) # Para evitar de um usário mandar ter menos que 1 sample por nó 
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
        self.values_per_leaf = None#maybe erase 
        self.aux_purity_gain = None
        self.max_leaf = max_leaf
        
    def tree_structure(self,X):
        if self.max_depth:
            n = (2 **(self.max_depth +1)) - 1 
        else:
            N = np.shape(X)[0]
            n = 2 * np.floor(N/self.min_samples_leaf) -1 # maybe change to ceil to be safe that it fits
        self.children_left = np.ones(n) * -1 
        self.children_right = np.ones(n) * -1 
        self.feature = np.ones(n) * -1 
        self.node_depth = np.ones(n) * -1 #impossible, so that is ok 
        self.threshold = np.full(n, np.nan)
        self.impurity = np.ones(n) * -1 
        self.value = np.full(n, np.nan)
        self.purity_gain = np.ones(n) * -1 
        self.start = np.ones(n) * -1
        self.end = np.ones(n) * -1
        self.values_per_leaf = np.arange(n)
        self.aux_purity_gain = np.ones(n) * -1 
        if self.max_leaf:
            self.max_leaf = self.max_leaf
            if self.max_leaf <1:
                raise ValueError("WYou must have at least one leaf node")
        else:
            self.max_leaf = np.floor(N/self.min_samples_leaf)  # talvez tenha um +1 ou -1 ai, sei la. Mas eh o maximo de folhas que poderiamos ter. Como eu tenho o minimo de nos por folha...

    def splitter(self, X,y,node,defenitive = False):

        matri_list = [] 
        possible_cuts = [] 
        y_list = [] 

        for predictor in range(y):

            X_c_p = X.copy()
            lines = X[self.start[node]:self.end[node]]
            ordered_ind = np.argsort(lines[:,predictor])
            final_lines = lines[ordered_ind]

            X_c_p [self.start[node]:self.end[node]] = final_lines

            y_c_p = y.copy()
            y_lines = y[self.start[node]:self.end[node]]
            final_y_lines = y_lines[ordered_ind]
            y_c_p [self.start[node]:self.end[node]] = final_y_lines

            #IMPORTANT : WE HAVE TO CHANGE THE ORDERS FROM Y AS WELL!! Look below:

            median = (final_lines[:-1] + final_lines[1:])/2 # Candidated for cut value. This will have to be chenged for classification (probably)
            m = 0
            gratest_purity_gain_per_p = None
            associated_value_per_p = None
            new_end_begin = None
            imp_left = None
            imp_right = None
            prev_l= None
            prev_r = None
            while m < len(median):
                #since we are using an ordinated array, we can use a function that uses binary search to avoid recounting repeated medians.
                id_last_m = np.searchsorted(median, median[m], side = 'right') - 1
                if median[m] == median[id_last_m]:
                    m = id_last_m 
                prevision_left = np.mean(y[self.start[node]:self.start[node]+m]) # maybe we should add +1 
                prevision_right = np.mean(y[self.start[node]+m:self.end[node]]) # maybe we should add +1 in somewhere here too. Or +2 

                '''
                
                    Now we have to compare the gain of purity. Since we have got a way of limiting the minimum amout of 
                    samples in a leaf ( even though the standard is only one), we would have to compare those stuff. If it was 
                    the simple case ( equals to one) then e would be sure that it is okay ( becase it is impossible to be lower).
                
                    I fell that this step could be more optimal
                                
                                
                    '''
                
                if m+1 >=  self.min_samples_leaf and len(median) - (m +1) >= self.min_samples_leaf:
                    left_mse = MSE(y[self.start[node]:m],prevision_left) 
                    right_mse = MSE(y[m:self.end[node]],prevision_right)
                    current_purity_gain = self.impurity[node] - ((m+1)/ len(median) * left_mse + (len(median)-(m+1))/len(median)*right_mse)
                    if current_purity_gain >= self.min_impurity_decrease:
                        if  gratest_purity_gain_per_p:
                            if gratest_purity_gain_per_p >current_purity_gain:
                                gratest_purity_gain_per_p = current_purity_gain
                                associated_value_per_p = median[m]
                                new_end_begin = m 
                                imp_left = left_mse
                                imp_right = right_mse
                                prev_l= prevision_left
                                prev_r = prevision_right
                        else:
                            gratest_purity_gain_per_p = current_purity_gain
                            associated_value_per_p = median[m]
                            new_end_begin = m #carefull
                            imp_left = left_mse
                            imp_right =right_mse
                            prev_l= prevision_left
                            prev_r = prevision_right
                m +=1
            if gratest_purity_gain_per_p:
                matri_list.append(X_c_p)
                possible_cuts.append([gratest_purity_gain_per_p,associated_value_per_p,predictor,new_end_begin,imp_left, imp_right,prev_l,prev_r])
                y_list.append(y_c_p)

        y_list = np.array(y_list)
        matri_list = np.ndarray(matri_list)
        possible_cuts = np.ndarray(possible_cuts)


        if y_list.shape[0] == 0: 
            return None 
        else:
            gratest_purity_gain_overall_index = np.argmax(possible_cuts[:,0])
            gratest_purity_gain_overall = possible_cuts[gratest_purity_gain_overall_index,0]
            if defenitive == False:
                return gratest_purity_gain_overall
            else:
                self.threshold[node] = possible_cuts[gratest_purity_gain_overall_index,1]
                self.feature[node] = possible_cuts[gratest_purity_gain_overall_index,2]
                r1 = np.max(self.children_right)
                r2 = np.max(self.children_left)
                right = max(r1,r2)+1
                left = right+1
                self.children_left[node] = left
                self.children_right[node] = right
                self.value[left] = possible_cuts[gratest_purity_gain_overall_index,6]
                self.impurity[left] = possible_cuts[gratest_purity_gain_overall_index,4]
                self.start[left] = self.start[node]
                self.end[left] = possible_cuts[gratest_purity_gain_overall_index,3]
                self.node_depth[left] = self.node_depth[node] +1 

                if (self.end[left] - self.start[left]) >= self.min_samples_split and self.node_depth[left] > self.max_depth and self.impurity[left] > epsolon:
                    self.aux_purity_gain[left] = 1



                self.value[right] = possible_cuts[gratest_purity_gain_overall_index,7]
                self.impurity[right] = possible_cuts[gratest_purity_gain_overall_index,5]
                self.start[right]= possible_cuts[gratest_purity_gain_overall_index,3] #might need to review that
                self.end[right] = self.end[node]
                self.node_depth[right] = self.node_depth[node] +1 

                if (self.end[right] - self.start[right]) >= self.min_samples_split and self.node_depth[right] > self.max_depth and self.impurity[right] > epsolon:
                                    self.aux_purity_gain[right] = 1
                return gratest_purity_gain_overall # do not think is necessary here, but okay          


    def tree_construction(self, X,y):
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        self.tree_structure(X)
        self.value[0] = np.mean(y) # considering just regression. Might be generlized later
        self.impurity[0] = MSE(y,self.value[0]) # considering just regression. Might be generlized later
        self.start[0] = 0
        self.end[0] = X.shape[0] # considerind [start , end)
        self.node_depth[0] = 0

        list_nodes_that_can_be_split =  [] # since we can not import

        if self.max_leaf >1 and  X.shape[0] >=  self.min_samples_split and self.max_depth >=1 and self.impurity[0]>epsolon:
            gain = self.splitter(X,y,0)
            if gain:
                list_nodes_that_can_be_split.append([gain,0])

    
        while ( 1 in self.aux_purity_gain and self.max_splits>0):
            if len(list_nodes_that_can_be_split) >0:
                chosen = max(list_nodes_that_can_be_split, key = lambda x:x[0]) 
                node_chosen_to_split = chosen[1]
                self.aux_purity_gain[node_chosen_to_split] = -1 # turning it back to -1 
                u = self.splitter(X,y,node_chosen_to_split, defenitive= True)
                self.max_splits -=1
                if self.aux_purity_gain[self.children_left[node_chosen_to_split]] == 1: 
                    new_candidate = self.splitter(X,y,node_chosen_to_split)
                    list_nodes_that_can_be_split.append([new_candidate,self.children_left[node_chosen_to_split]])
                if self.aux_purity_gain[self.children_right[node_chosen_to_split]] == 1: 
                                    new_candidate = self.splitter(X,y,node_chosen_to_split)
                                    list_nodes_that_can_be_split.append([new_candidate,self.children_right[node_chosen_to_split]])
    def predict (self,X):

        y_num = X.shape[0]

        y = [] 

        for i in range(y_num):
            node = 0
            while self.children_left[node] != -1: # Becuase if it has left child, then it has right child
                if X[i][self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_left[node]
            pred_y = self.value[node]
            y.append(pred_y)
        y = np.ndarray(y)
        return y
                     
        






    





class RegressionTree(Regression):
    def __init__(self): 
        super.__init__()
    def set_params(self, min_impurity_decrease = 0.1,max_depth = None):
        self.min_impurity_decrease = min_impurity_decrease 
        self.max_depth = max_depth
    def fit(self,X,Y):
        super().tree_construction(X,Y)
        self.prune()
    def prune():
        pass
        
    
         
         


class Bagging_Base():
    pass

class BaggingRegression(Regression, Bagging_Base):
    def __init__(self):
         super.__init__()

    
    pass

