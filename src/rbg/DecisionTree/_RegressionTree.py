import numpy as np
from rbg.Utilities import MSE, rss, RMSE, MAE, RSquare
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode


class Regression:
    '''

    Initializes the metrics used to calculate the impurity (that is, the metrics to split the nodes),
    predict and evaluate. For the impurity, it allows the user to use MSE as default, but also allows the regression 
    metrics implemented by the library rss, RMSE, MAE and RSquare. The user an also define a function 
    of his choice, however, it must take as parameters, y_predicted and y_true, respectively.
    For the predict, it allows mean (the standard), median, max, min, mode, or another one chosen by the
    user. However, the user must define and submit a function that accepts an array as a paramter, and 
    "axis", if the user is performing bagging, random forest or prunning a tree. 
    The evaluation function is by standard "Same", to indicate that is the same as the impurity metric. 
    The user is allowed to choose another one, following the same rules of either selecting rss, RMSE, MAE,
    RSquare or to define a function of his choice,that must take as parameters, y_predicted and y_true, respectively.

    Args:

        criterion_split: string defining what will be the function to calculate the impurity (if "Other", then the user must provide a function)
        custom_split: the function itself (None if the user choose one of the MSE, rss, RMSE, MAE or RSquare)    
        criterion_choose: string defining what will be the function to select the y_predic value (if "Other", then the user must provide a function)
        custom_choose: the function itself (None if the user choose one of the mean. median, min, max or mode) 
        criterion_evaluation: string defining what will be the function to calculate the evluation / error (if "Other", then the user must provide a function)
        custom_evaluation: the function itself (None if the user choose one of the MSE, rss, RMSE, MAE or RSquare) 

    Raises:

        TypeError: If the user assing anything but a string to criterion_split, criterion_choose or criterion_evaluation
        ValueError: if any of the strings from criterion_split, criterion_choose or criterion_evaluation are invalid strings
        Warning:If the user chooses a different metric for the evaluation from the impurity metric. 
    '''
    def __init__(self,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Same", custom_evaluation = None):
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
        self.custom_split  = split.get(criterion_split)

        y_pred ={
            'Standard': np.mean,
            'mean':np.mean,
            'median':np.median,
            'max':np.max,
            'min':np.min,
            'mode': mode,
            'Other':custom_choose
        }
        if criterion_choose not in y_pred:
            raise ValueError ("The only split criterions avaiable for assinging y_pred are mean, mode, median, min, and max")
        self.custom_choose  = y_pred.get(criterion_choose)
        evalu ={
            'MSE':MSE,
            'RSS':rss,
            'RMSE':RMSE,
            'MAE':MAE,
            'RSquare': RSquare,
            "Same": self.custom_split,
            'Other':custom_evaluation
        } 
        if criterion_evaluation not in evalu:
            raise ValueError ("The only evalu criterions avaiable for evaluating the tree are mean, MSE, RSS, RMSE, MAE, RSquare, the Same as used in spliting and Other")
        self.custom_evaluation = evalu.get(criterion_evaluation)
        if criterion_evaluation !="Same":
            raise Warning("It is possible, but probably inefficient to assaing a different method of evaluation")
        self.criterion_split = criterion_split
        self.criterion_choose = criterion_choose
        self.criterion_evaluation = criterion_evaluation