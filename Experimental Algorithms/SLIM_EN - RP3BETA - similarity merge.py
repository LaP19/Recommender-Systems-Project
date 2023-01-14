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
URM_path = '/kaggle/input/dataset/interactions_and_impressions.csv'
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
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# os.environ["PYTHONWARNINGS"] = ('ignore::exceptions.ConvergenceWarning:sklearn.linear_model')
# os.environ["PYTHONWARNINGS"] = ('ignore:Objective did not converge:ConvergenceWarning:')

class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

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


recommender_SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
recommender_SLIMElasticNet.fit(epochs = 700, l1_ratio=0.049999999999999996, alpha = 0.001, positive_only = True, topK = 1000)

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

recommender_RP3beta = RP3betaRecommender(URM_train)
recommender_RP3beta.fit(alpha = 0.13342034459968835, beta = 0.31807035025857244, topK = 65, implicit = True)

from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

# ItemKNNCustomSimilarityRecommender is a recommender that supports a costum similary

"""
for alpha in np.arange(0.0, 1.1 , 0.1):
    print("alpha = ",alpha)
    new_similarity = (1 - alpha) * recommender_SLIMElasticNet.W_sparse + alpha * recommender_RP3beta.W_sparse

    recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
    recommender_object.fit(new_similarity)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
    print("MAP = ",result_df["MAP"])
    print("---------")



for alpha in np.arange(0.35, 0.45 , 0.01):
    print("alpha = ",alpha)
    new_similarity = (1 - alpha) * recommender_SLIMElasticNet.W_sparse + alpha * recommender_RP3beta.W_sparse

    recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
    recommender_object.fit(new_similarity)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
    print("MAP = ",result_df["MAP"])
    print("---------")

"""
#Train the recommenders separately and then merge

recommender_SLIMElasticNet = SLIMElasticNetRecommender(URM_all)
recommender_SLIMElasticNet.fit(epochs=700, l1_ratio=0.049999999999999996, alpha=0.001, positive_only=True, topK=1000)

recommender_RP3beta = RP3betaRecommender(URM_all)
recommender_RP3beta.fit(alpha=0.13342034459968835, beta=0.31807035025857244, topK=65, implicit=True)

# best model - alpha = 0.39
alpha = 0.39
new_similarity = (1 - alpha) * recommender_SLIMElasticNet.W_sparse + alpha * recommender_RP3beta.W_sparse

recommender = ItemKNNCustomSimilarityRecommender(URM_all)
recommender.fit(new_similarity)

test_users = pd.read_csv('/kaggle/input/newdata/data_target_users_test.csv')
test_users

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