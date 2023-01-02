#Best one so far --> score: 0.05698
#Part of the code is taken by the practice of Dacrema on Slim with Elastic Net

#To run this code in kaggle, after having imported the recsys course code in the notebook you have to
# write these two lines at the beginning:
# !cp -r ../input/utilscode/* ./
# !python run_compile_all_cython.py
# And delete the os.system() lines
import pandas as pd
import numpy as np
from numpy.ma import MaskedArray
import sklearn.utils.fixes
import scipy.sparse as sps

import warnings
warnings.filterwarnings("ignore")

sklearn.utils.fixes.MaskedArray = MaskedArray

import os
os.system(r"run_compile_all_cython.py")

# Load the data
URM_path = 'Data/interactions_and_impressions.csv'
URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                sep=",",
                                header=0, engine='python')
URM_all_dataframe.columns = ["UserID", "ItemID", "Impressions", "Data"]

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

# Build the URM: I turn every kind of interaction as a 1, so I first eliminate all the duplicate and then turn every
# value remained in the Data of the iteractions_and_impressions into a 1
URM_all_dataframe = URM_all_dataframe.drop_duplicates(['UserID', 'ItemID'], keep='first')

URM_all = sps.coo_matrix((np.ones(len(URM_all_dataframe["Data"].values)),
                          (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)))
URM_all = URM_all.tocsr()  # to obtain fast access to rows (users)

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

# split data into train and validation data
URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.80)

from Evaluation.Evaluator import EvaluatorHoldout

# create an evaluator object to evaluate validation set
# will use it for hyperparameter tuning
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

import numpy as np
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    #initialize the object
    def __init__(self, URM_train, verbose=True):
        super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose=verbose)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha=1.0, positive_only=True, topK=100, **earlystopping_kwargs):

        assert l1_ratio >= 0 and l1_ratio <= 1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(
            self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)
        # check_matrix takes a matrix as input and transforms it into the specified format
        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if time.time() - start_time_printBatch > 300 or currentItem == n_items - 1:
                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem) / elapsed_time))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

#recommender used
recommender_class = SLIMElasticNetRecommender
import os

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

#start Hyperparameter tuning
n_cases = 50
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10
from skopt.space import Real, Integer, Categorical


hyperparameters_range_dictionary = {
    "l1_ratio": Real(low = 0.001, high = 0.05, prior = 'log-uniform'), #prior = log-uniform means that valeus
                                                # are sampled uniformly between log(lower, base) and log(upper, base)
                                                # (default base is 10)
    "alpha": Real(low = 0.001, high = 0.1, prior = 'log-uniform'), #low and high are the lower bound and the upper bound
    "positive_only": Categorical([True]),
    "topK": Integer(500,1000)
}
#Setup the early stopping --> to save a lot of computational time
earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 5,
                          "validation_metric": metric_to_optimize,
                          }

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

#create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)

from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

#provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs
)
recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train_validation],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs
)
#let's run the bayesian search
hyperparameterSearch.search(recommender_input_args = recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "last",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )

from Recommenders.DataIO import DataIO

#explore the results of the search
data_loader = DataIO(folder_path = output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

search_metadata.keys()
hyperparameters_df = search_metadata["hyperparameters_df"]
result_on_validation_df = search_metadata["result_on_validation_df"]
best_hyperparameters = search_metadata["hyperparameters_best"]

# let's fit the model with the hyperparamethers obtained from the previous search and evaluate them on validation set

recommender = SLIMElasticNetRecommender(URM_all)
#after the hyperparameter tuning I found that the best hyperparameters configuration is
# (l1_ratio=0.001, alpha=0.01, positive_only=True, topK=450), so I fit the recommender with these values
recommender.fit(epochs = 700, l1_ratio=0.049999999999999996, alpha = 0.001, positive_only = True, topK = 1000)

recommender.save_model(output_folder_path, file_name=recommender.RECOMMENDER_NAME + "_my_own_save.zip")

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
test_users.to_csv('submission.csv', index=False)