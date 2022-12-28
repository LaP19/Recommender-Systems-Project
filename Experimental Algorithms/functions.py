import Data_manager.split_functions.split_train_validation_random_holdout as split
import numpy as np


"""
@author: Valeria Amato
"""
def split_train_in_three_percetanges(URM_all, user_wise, validation_percetage, test_percentage):
    """
    The function splits an URM in three matrices selecting the number of interactions globally
    if the user wise parameter is False, otherwise selecting the number of interactions one user at a time
    :param URM_all:
    :user_wise: 
    :param validation_percentage:
    :parm test_percetage
    :return:
    """
    train_percetage = 1 - validation_percetage - test_percentage
    real_val_percetage = validation_percetage / (validation_percetage + test_percentage)

    if user_wise:
        URM_train, URM_rest = split.split_train_in_two_percentage_user_wise(URM_all, train_percetage)
        URM_validation, URM_test = split.split_train_in_two_percentage_user_wise(URM_rest, real_val_percetage)
    else:
        URM_train, URM_rest = split.split_train_in_two_percentage_global_sample(URM_all, train_percetage)
        URM_validation, URM_test = split.split_train_in_two_percentage_global_sample(URM_rest, real_val_percetage)

    return URM_train, URM_validation, URM_test

def precision(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    
    return precision_score

def recall(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

def AP(recommended_items, relevant_items):
   
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return ap_score