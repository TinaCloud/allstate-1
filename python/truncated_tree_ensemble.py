__author__ = 'brandonkelly'

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.base import clone
import os
import multiprocessing
import time
import cPickle
import matplotlib.pyplot as plt

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'

njobs = multiprocessing.cpu_count() - 1
njobs = 3


def get_truncated_shopping_indices(n_shop, p=0.3):
    """
    Return the indices of the stacked array (customer_ID, shopping point) for a randomly truncated shopping history.

    :param n_shop: The number of shopping points for each customer.
    :param p: The probability of for truncation following a geometric distribution.
    :return: The indices of the stacked array.
    """
    n_trunc = np.random.geometric(p, len(n_shop)) + 1
    n_trunc = np.where(n_trunc > n_shop, n_shop, n_trunc)
    first_spt_idx = np.roll(np.cumsum(n_shop), 1)
    first_spt_idx[0] = 0
    # need the minus one here since the shopping points index starts at one and n_trunc starts at 2
    idx = first_spt_idx + n_trunc - 1

    return idx


class TruncatedHistoryTrees(object):
    def __init__(self, ntrees, criterion="gini", splitter="best", max_depth=None, max_features=None, verbose=False):
        self.ntrees = ntrees
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.verbose = verbose
        self.base_estimator = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                                     max_depth=self.max_depth, max_features=self.max_features)
        self.trees = []
        self.nclasses = 0

    def fit(self, X, y, n_shop):
        classes, y = np.unique(y, return_inverse=True)
        self.classes = classes
        for i in xrange(self.ntrees):
            if self.verbose:
                print 'Fitting tree', i, 'out of', self.ntrees
            # get the indices
            trunc_idx = get_truncated_shopping_indices(n_shop)
            X_trunc = X[trunc_idx]
            this_tree = clone(self.base_estimator)
            this_tree.fit(X_trunc, y)
            self.trees.append(this_tree)

        return self

    def predict(self, X, majority_vote=True):
        if majority_vote:
            y_votes = np.zeros((X.shape[0], len(self.classes)))
            for tree in self.trees:
                vote = tree.predict(X)
                idx_1d = vote + np.arange(len(vote)) * y_votes.shape[1]
                # add vote for each class
                y_votes[np.unravel_index(idx_1d, y_votes.shape)] += 1

            y_predict = self.classes[y_votes.argmax(axis=1)]  # output is winner of majority vote

        else:
            # predict class by averaging probabilities
            yprob = np.zeros((X.shape[0], len(self.classes)))
            for tree in self.trees:
                yprob += tree.predict_proba(X)

            yprob /= self.ntrees
            # classify based on highest probability from tree ensemble
            class_idx = yprob.argmax(axis=1)
            y_predict = self.classes[class_idx]

        return y_predict


def validate_trunc_tree(args):
    X_train, y_train, X_val, y_val, n_shop, grid = args

    cv_score = np.zeros(len(grid))

    print 'Calculating CV scores for tuning parameters:'
    for j, params in enumerate(grid):
        print j, params
        trunc_tree = TruncatedHistoryTrees(ntrees, **params).fit(X_train, y_train, n_shop)
        y_predict = trunc_tree.predict(X_val)
        cv_score[j] = accuracy_score(y_val, y_predict)

    return cv_score


def fit_truncated_tree(training_set, response, test_set, ntrees, n_shop):

    pool = multiprocessing.Pool(njobs)
    pool.map(int, range(njobs))

    test_cust_ids = test_set.index.get_level_values(0).unique()
    train_cust_ids = training_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), 3), dtype=np.int), index=test_cust_ids,
                             columns=['MajorityVote', 'AverageProb', 'LastObsValue'])

    # get CV split
    folds = KFold(len(train_cust_ids), n_folds=njobs)

    # run over a grid of tuning parameters and choose the values that minimize the CV score
    pgrid = {'max_features': list(np.unique(np.logspace(np.log10(2),
                                                        np.log10(training_set.shape[1]), 5).astype(np.int))),
             'max_depth': [5, 10, 20, 50, None]}
    grid = list(ParameterGrid(pgrid))

    cv_args = []

    training_set = training_set.reset_index(level=1)

    print 'Using freatures', training_set.columns

    for train, validate in folds:
        # reset the index here to include the shopping point as a predictor
        X_train = training_set.ix[train_cust_ids[train]].values
        y_train = response.ix[train_cust_ids[train]].values
        n_shop_train = n_shop[train]
        X_val = training_set.ix[train_cust_ids[validate]].values
        y_val = response.ix[train_cust_ids[validate]].values

        # X_val contains the entire shopping history, so we need to truncate it so it is more representative of the
        # test set
        n_shop_val = n_shop[validate]
        trunc_idx = get_truncated_shopping_indices(n_shop_val)
        X_val = X_val[trunc_idx]

        cv_args.append((X_train, y_train, X_val, y_val, n_shop_train, grid))

    cv_scores = pool.map(validate_trunc_tree, cv_args)
    cv_score = np.zeros(len(cv_scores[0]))
    for score in cv_scores:
        cv_score += score
    cv_score /= len(cv_scores)

    # print 'Individual CV scores:'
    # for k in range(len(cv_score)):
    #     print grid[k], cv_score[k]
    #
    # print ''

    best_idx = cv_score.argmax()
    best_params = grid[best_idx]
    print 'Best tree parameters are:'
    print best_params
    print 'with a validation accuracy score of', cv_score[best_idx]

    print 'Refitting model...'
    X = training_set.values
    y = response.values
    trunc_tree = TruncatedHistoryTrees(ntrees, **best_params).fit(X, y, n_shop)

    # predict the test data
    print 'Predicting the class for the test set...'
    # need to get indices for last observed value
    n_shop_test = np.asarray([test_set.ix[cid].index[-1] for cid in test_cust_ids])
    first_spt_idx = np.roll(np.cumsum(n_shop_test), 1)
    first_spt_idx[0] = 0
    # need the minus one here since the shopping points index starts at one and n_trunc starts at 2
    last_idx = first_spt_idx + n_shop_test - 1
    X_test = test_set.reset_index(level=1).values
    X_test = X_test[last_idx]
    y_predict['MajorityVote'] = trunc_tree.predict(X_test)
    y_predict['AverageProb'] = trunc_tree.predict(X_test, majority_vote=False)

    return y_predict, trunc_tree


if __name__ == "__main__":

    ntrees = 500

    training_set = pd.read_hdf(data_dir + 'training_set.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set.h5', 'df')

    # for testing, reduce number of customers
    # customer_ids = training_set.index.get_level_values(0).unique()
    # training_set = training_set.select(lambda x: x[0] < customer_ids[1000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    response_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    response = training_set[training_set['record_type'] == 1][response_columns]
    response = response.reset_index(level=1).drop(training_set.index.names[1], axis=1)

    training_set = training_set[training_set['record_type'] == 0]
    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID',
                      'most_common_plan', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    last_spt_idx = [(cid, test_set.ix[cid].index[-1]) for cid in customer_ids]

    predictions = []
    for category in response_columns:
        print 'Training category', category, '...'
        tstart = time.clock()
        this_prediction, trunc_tree = fit_truncated_tree(training_set, response[category], test_set, ntrees, n_shop)
        this_prediction['LastObsValue'] = test_set.ix[last_spt_idx][category]

        print 'Pickling models...'
        cPickle.dump(trunc_tree, open(data_dir + 'models/truncated_trees_category' + category + '.pickle', 'wb'))

        predictions.append(this_prediction)
        tend = time.clock()
        print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

    predictions = pd.concat(predictions, axis=1, keys=response_columns)
    predictions.to_hdf(data_dir + 'truncated_trees_independent_predictions.h5', 'df')