import Data_manager.split_functions.split_train_validation_random_holdout as split

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
