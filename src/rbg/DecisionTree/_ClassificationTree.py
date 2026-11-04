import numpy as np
from rbg.Utilities import MSE, rss, RMSE, MAE, RSquare
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode


class Classification:
    def __init__(self,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard",custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None):
        split ={
            'Standard':Gini,
            'Gini':Gini,
            'classif_error':classif_error,
            'entropy':entropy,
            'Other':custom_split
        }
        if criterion_split not in split:
            raise ValueError ("The only split criterions avaiable for classification are Gini,classif_error,entropy, and Other")
        self.custom_split  = split.get(criterion_split)

        y_pred ={
            'Standard': mode,
            'median':np.median,
            'max':np.max,
            'min':np.min,
            'mode':mode,
            'Other':custom_choose 
        }
        if criterion_choose not in y_pred:
            raise ValueError ("The only split criterions avaiable for assinging y_pred are mode, median, min, and max")
        self.custom_choose  = y_pred.get(criterion_choose)
        evalu ={
            'Standard':Gini,
            'Gini':Gini,
            'classif_error':classif_error,
            'entropy':entropy,
            'Same':self.custom_split,
            'Other':custom_evaluation
        
        } 

        if criterion_evaluation not in evalu:
            raise ValueError ("The only evaluation criterions avaiable for evaluating the tree are Gini, entropy, criterion_split, the Same as used in spliting and Other")
        self.custom_evaluation  = evalu.get(criterion_evaluation)
        if criterion_evaluation !="Same":
            raise Warning("It is possible, but probably inefficient to assaing a different method of evaluation")

        self.criterion_split = criterion_split
        self.criterion_choose = criterion_choose
        self.criterion_evaluation = criterion_evaluation
