__author__ = 'brandonkelly'

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.base import _partition_estimators
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.base import clone
import os
import multiprocessing
import time
import cPickle
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Parallel, delayed
import itertools
from truncated_tree_ensemble import get_truncated_shopping_indices

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'

njobs = multiprocessing.cpu_count() - 1
njobs = 3
verbose = 2

MAX_INT = np.iinfo(np.int32).max


def _parallel_trees_prediction(base_tree, ntrees, X, y, n_shop, X_test, verbose):
    """Private function used to provide class predictions for a batch of trees within a job."""
    y_votes = np.zeros((X_test.shape[0], len(np.unique(y))), dtype=np.uint16)
    for i in xrange(ntrees):
        if verbose > 1:
            print("building tree %d of %d" % (i + 1, ntrees))
        # get the indices of the truncated shopping history
        trunc_idx = get_truncated_shopping_indices(n_shop)
        X_trunc = X[trunc_idx]
        tree = clone(base_tree)
        tree.fit(X_trunc, y, check_input=False)
        vote = tree.predict(X_test)
        idx_1d = vote + np.arange(len(vote)) * y_votes.shape[1]
        # add vote for each class
        y_votes[np.unravel_index(idx_1d, y_votes.shape)] += 1

    return y_votes


def fit_and_predict_test_plans(training_set, response, test_set, ntrees, n_shop, max_features):

    test_cust_ids = test_set.index.get_level_values(0).unique()
    training_set = training_set.reset_index(level=1)

    # need to get indices for last observed value
    n_shop_test = np.asarray([test_set.ix[cid].index[-1] for cid in test_cust_ids])
    first_spt_idx = np.roll(np.cumsum(n_shop_test), 1)
    first_spt_idx[0] = 0
    # need the minus one here since the shopping points index starts at one and n_trunc starts at 2
    last_idx = first_spt_idx + n_shop_test - 1
    test_set = test_set.reset_index(level=1)
    X_test = test_set.iloc[last_idx].values

    del test_set  # free memory

    # create mapping between original classes and unique set of class labels
    classes, y = np.unique(response.values, return_inverse=True)

    # Assign chunk of trees to jobs
    # Partition estimators between jobs
    n_estimators = (ntrees // njobs) * np.ones(njobs, dtype=np.int)
    n_estimators[:ntrees % njobs] += 1
    starts = np.cumsum(n_estimators)
    starts = [0] + starts.tolist()

    base_tree = DecisionTreeClassifier(max_features=max_features)

    # Parallel loop: we use the threading backend as the Cython code for
    # fitting the trees is internally releasing the Python GIL making
    # threading always more efficient than multiprocessing in that case.
    print 'Building the trees...'
    all_votes = Parallel(n_jobs=njobs, verbose=verbose,
                         backend="threading")(
        delayed(_parallel_trees_prediction)(
            base_tree,
            starts[i+1] - starts[i],
            training_set.values,
            y,
            n_shop,
            X_test,
            verbose)
        for i in range(njobs))

    y_votes_all = np.zeros_like(all_votes[0])

    for vote in all_votes:
        y_votes_all += vote

    del all_votes  # free memory

    y_predict = classes[y_votes_all.argmax(axis=1)]  # output is winner of majority vote

    # store in dataframe
    y_predict = pd.DataFrame(data=np.column_stack((y_predict, np.zeros(len(test_cust_ids), dtype=np.uint16))),
                             index=test_cust_ids, columns=['MajorityVote', 'LastObsValue'])

    return y_predict


if __name__ == "__main__":

    ntrees = 1000
    max_features = 20

    training_set = pd.read_hdf(data_dir + 'training_set.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set.h5', 'df')

    # for testing, reduce number of customers
    # customer_ids = training_set.index.get_level_values(0).unique()
    # print 'Reducing training set...'
    # select_customers = lambda x: x[0] < customer_ids[2000]
    # training_set = training_set.select(select_customers, axis=0)
    # customer_ids = test_set.index.get_level_values(0).unique()
    # print 'Reducing test set...'
    # test_set = test_set.select(select_customers, axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    response = training_set[training_set['record_type'] == 1]['planID']
    response.index = response.index.get_level_values(0)

    training_set = training_set[training_set['record_type'] == 0]
    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])

    last_spt_idx = [(cid, test_set.ix[cid].index[-1]) for cid in test_set.index.get_level_values(0).unique()]
    last_plan = test_set.ix[last_spt_idx]['planID']
    last_plan.index = last_plan.index.get_level_values(0)

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID',
                      'most_common_plan', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    tstart = time.clock()
    prediction = fit_and_predict_test_plans(training_set, response, test_set, ntrees, n_shop,
                                                        max_features)
    tend = time.clock()
    print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

    # helpful to compare with as a sanity check
    prediction['LastObsValue'] = last_plan

    print 'Writing predictions...'
    prediction.to_hdf(data_dir + 'truncated_forest_predictions_ntrees' + str(ntrees) + '_max_featuers' +
                      str(max_features) + '.h5', 'df')
