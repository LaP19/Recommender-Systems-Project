#This returns my best score: 0.046..

import pandas as pd
import numpy as np
from numpy.ma import MaskedArray
import sklearn.utils.fixes
import scipy.sparse as sps

sklearn.utils.fixes.MaskedArray = MaskedArray

import os
os.system(r"run_compile_all_cython.py")

# Load the data
URM_path = 'Data/interactions_and_impressions.csv'
URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                sep=",",
                                header=0, engine='python')
URM_all_dataframe.columns = ["UserID", "ItemID", "Impressions", "Data"]

episodes_path = 'Data/data_ICM_length.csv'
episodes_all_dataframe = pd.read_csv(filepath_or_buffer=episodes_path,
                                     sep=",",
                                     header=0, engine='python')
episodes_all_dataframe.columns = ["ItemID", "FeatureID", "Data"]

array_length_episodes = [0] * 27967
for i in range(23090):
    num = episodes_all_dataframe._get_value(i, "ItemID", takeable=False)
    array_length_episodes[num] = episodes_all_dataframe._get_value(i, "Data", takeable=False)

print("The number of interactions is {}".format(len(URM_all_dataframe)))

userID_unique = URM_all_dataframe["UserID"].unique()
itemID_unique = URM_all_dataframe["ItemID"].unique()

n_users = len(userID_unique)
n_items = len(itemID_unique)
n_interactions = len(URM_all_dataframe)

print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemID_unique), max(userID_unique)))
print("Average interactions per user {:.2f}".format(n_interactions / n_users))
print("Average interactions per item {:.2f}\n".format(n_interactions / n_items))

print("Sparsity {:.2f} %".format((1 - float(n_interactions) / (n_items * n_users)) * 100))

# Build the URM: 2-URM_all_dataframe["Data"].values is because I want to give more imnce to the Data that is 0 (the
# user has seen the episode) and slightly less importance to the Data that is 1 (the user has read the details)
URM_all = sps.coo_matrix((2 - URM_all_dataframe["Data"].values,
                          (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)), dtype=np.float64)

URM_all.tocsr()


from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

# split data into train and validation data 85/15
URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.85)

from Evaluation.Evaluator import EvaluatorHoldout

# create an evaluator object to evaluate validation set, I will use it for hyperparameter tuning
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

# try a simple CF model based on item-item similarity, here just select the class of model to perform tuning
recommender_class = ItemKNNCFRecommender

from skopt.space import Integer, Categorical

# Choose possible values for the hyperparameters (random search). ItemKNNCF uses topK (K param), shrink term (to
# consider support of similarity), similarity type (cosine), normalization of data (true, false)
hyperparameters_range_dictionary = {
    "topK": Integer(5, 1000),
    "shrink": Integer(0, 1000),
    "similarity": Categorical(["cosine"]),
    "normalize": Categorical([True, False]),
    "feature_weighting": Categorical(["TF-IDF"])
}

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

# Use Baesyan Search --> uses Gaussian process to model interdependencies between hyperparameters based on how they
# affect the result. Pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={}
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={}
)
import os

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# I choose 50 cases with 30% random
n_cases = 50
n_random_starts = int(n_cases * 0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

# let's run the bayesian search
hyperparameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            hyperparameter_search_space=hyperparameters_range_dictionary,
                            n_cases=n_cases,
                            n_random_starts=n_random_starts,
                            save_model="last",
                            output_folder_path=output_folder_path,  # Where to save the results
                            output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                            metric_to_optimize=metric_to_optimize,
                            cutoff_to_optimize=cutoff_to_optimize,
                            )

from Recommenders.DataIO import DataIO

# explore the results of the search. The metadata.zip file contains details on the search
data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

search_metadata.keys()

hyperparameters_df = search_metadata["hyperparameters_df"]

result_on_validation_df = search_metadata["result_on_validation_df"]

best_hyperparameters = search_metadata["hyperparameters_best"]

# let's fit the model with the hyperparamethers obtained from the previous search and evaluate them on validation set

recommender = ItemKNNCFRecommender(URM_all)
recommender.fit(shrink=999, topK=407, feature_weighting='TF-IDF', similarity='cosine', normalize=True)
# evaluator_valid.evaluateRecommender(recommender)

test_users = pd.read_csv('Data/data_target_users_test.csv')

user_id = test_users['user_id']
recommendations = []
for user in user_id:
    recommendations.append(recommender.recommend(user, cutoff=10))
for index in range(len(recommendations)):
    recommendations[index] = np.array(recommendations[index])

test_users['item_list'] = recommendations
test_users['item_list'] = pd.DataFrame(
    [str(line).strip('[').strip(']').replace("'", "") for line in test_users['item_list']])
test_users.to_csv('submission_new3.csv', index=False)
