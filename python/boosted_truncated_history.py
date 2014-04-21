__author__ = 'brandonkelly'

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import VerboseReporter
from sklearn.utils import check_arrays, check_random_state, column_or_1d
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.tree._tree import DTYPE, PresortBestSplitter, FriedmanMSE
from truncated_tree_ensemble import get_truncated_shopping_indices
from classify_plans_independently import LastObservedValue
from time import time
import pandas as pd
import os
import cPickle
import multiprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'

njobs = 7
ntrees = 1000


class TruncatedHistoryGBC(GradientBoostingClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=0.1, min_samples_split=2,
                 min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0):
        super(TruncatedHistoryGBC, self).__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                                  subsample=subsample, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_depth=max_depth, init=init,
                                                  random_state=random_state, max_features=max_features, verbose=verbose)
        self.classes_ = None
        self.n_classes_ = 0

    def _fit_stages(self, X, y, y_pred, random_state, begin_at_stage, n_shop):
        n_samples = len(n_shop)
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        # init criterion and splitter
        criterion = FriedmanMSE(1)
        splitter = PresortBestSplitter(criterion,
                                       self.max_features_,
                                       self.min_samples_leaf,
                                       random_state)

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        # perform boosting iterations
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            # first randomly truncate the shopping history
            trunc_idx = get_truncated_shopping_indices(n_shop)
            X_trunc = X[trunc_idx]

            sample_mask = _random_sample_mask(n_samples, n_inbag,
                                              random_state)
            # OOB score before adding this stage
            old_oob_score = loss_(y[~sample_mask],
                                  y_pred[~sample_mask])

            # fit next stage of trees
            y_pred = self._fit_stage(i, X_trunc, y, y_pred, sample_mask,
                                     criterion, splitter, random_state)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             y_pred[sample_mask])
                self._oob_score_[i] = loss_(y[~sample_mask],
                                            y_pred[~sample_mask])
                self.oob_improvement_[i] = old_oob_score - self._oob_score_[i]
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, y_pred)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

        return i + 1

    def fit_all(self, X, y, n_shop):
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense")
        y = column_or_1d(y, warn=True)
        n_samples, n_features = X.shape
        self.n_features = n_features
        random_state = check_random_state(self.random_state)
        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model
            idx = get_truncated_shopping_indices(n_shop)
            self.init_.fit(X[idx], y)

            # init predictions
            y_pred = self.init_.predict(X[idx])
            begin_at_stage = 0
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            y_pred = self.decision_function(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(X, y, y_pred, random_state, begin_at_stage, n_shop)
        # change shape of arrays after fit (early-stopping or additional tests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
            if hasattr(self, '_oob_score_'):
                self._oob_score_ = self._oob_score_[:n_stages]

        return self

    def fit(self, X, y, n_shop=None):
        try:
            n_shop is not None
        except ValueError:
            "Must supply n_shop."

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # initial pass through the data
        self.fit_all(X, y, n_shop)

        # find where the oob score is maximized, only use this many trees
        oob_score = np.cumsum(self.oob_improvement_)
        ntrees = oob_score.argmax() + 1
        if self.verbose:
            print 'Chose', ntrees, 'based on the OOB score.'
        self.n_estimators = ntrees
        self.estimators_ = self.estimators_[:ntrees]

        # plt.plot(oob_score)
        # plt.show()

        return self


def validate_gbc_tree(args):
    X_train, y_train, X_val, y_val, n_shop, grid, last_obs_idx = args

    cv_score = np.zeros(len(grid))

    print 'Calculating CV scores for tuning parameters:'
    for j, params in enumerate(grid):
        print j, params
        trunc_tree = TruncatedHistoryGBC(subsample=0.015, learning_rate=0.01, n_estimators=ntrees, max_depth=params,
                                         verbose=True)
        trunc_tree.fit(X_train, y_train, n_shop)
        y_predict = trunc_tree.predict(X_val)
        cv_score[j] = accuracy_score(y_val, y_predict)

    return cv_score


def fit_truncated_gbc(training_set, response, test_set, n_shop, category):
    pool = multiprocessing.Pool(njobs)
    pool.map(int, range(njobs))

    test_cust_ids = test_set.index.get_level_values(0).unique()
    train_cust_ids = training_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), 2), dtype=np.int), index=test_cust_ids,
                             columns=['TruncatedGBC', 'LastObsValue'])

    # get CV split
    folds = KFold(len(train_cust_ids), n_folds=njobs)

    # run over a grid of tuning parameters and choose the values that minimize the CV score
    grid = list([1, 2, 3, 4, 5])

    cv_args = []

    # include the shopping point as a predictor
    training_set = training_set.reset_index(level=1)

    print 'Using features', training_set.columns

    last_obs_idx = []
    for j, c in enumerate(training_set.columns):
        if 'is_last_value_' + category in c:
            last_obs_idx.append(j)

    print 'Columns corresponding to last observed value:', training_set.columns[last_obs_idx]

    for train, validate in folds:
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

        cv_args.append((X_train, y_train, X_val, y_val, n_shop_train, grid, last_obs_idx))

    cv_scores = pool.map(validate_gbc_tree, cv_args)
    cv_score = np.zeros(len(cv_scores[0]))
    for score in cv_scores:
        cv_score += score
    cv_score /= len(cv_scores)

    print 'Individual CV scores:'
    for k in range(len(cv_score)):
        print grid[k], cv_score[k]

    print ''

    best_idx = cv_score.argmax()
    best_params = grid[best_idx]
    print 'Best tree parameters are:'
    print best_params
    print 'with a validation accuracy score of', cv_score[best_idx]

    print 'Refitting model...'
    X = training_set.values
    y = response.values
    trunc_tree = TruncatedHistoryGBC(max_depth=best_params).fit(X, y, n_shop)

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
    y_predict['TruncatedGBC'] = trunc_tree.predict(X_test)

    pool.close()

    return y_predict, trunc_tree


if __name__ == "__main__":

    training_set = pd.read_hdf(data_dir + 'training_set.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set.h5', 'df')

    # for testing, reduce number of customers
    # customer_ids = training_set.index.get_level_values(0).unique()
    # training_set = training_set.select(lambda x: x[0] < customer_ids[1000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    response_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    response = training_set[training_set['record_type'] == 1][response_columns]
    response = response.reset_index(level=1).drop(training_set.index.names[1], axis=1)

    last_spt_idx = [(cid, test_set.ix[cid].index[-1]) for cid in test_set.index.get_level_values(0).unique()]
    last_observed_value = test_set.ix[last_spt_idx][response_columns].reset_index(level=1)

    training_set = training_set[training_set['record_type'] == 0]
    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID',
                      'most_common_plan', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    predictions = []
    for category in response_columns:
        print 'Training category', category, '...'
        tstart = time()
        this_prediction, trunc_gbc = fit_truncated_gbc(training_set, response[category], test_set, n_shop, category)
        this_prediction['LastObsValue'] = last_observed_value[category]

        print 'Pickling models...'
        cPickle.dump(trunc_gbc, open(data_dir + 'models/truncated_gbc_category' + category + '.pickle', 'wb'))

        predictions.append(this_prediction)
        tend = time()
        print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

    predictions = pd.concat(predictions, axis=1, keys=response_columns)
    predictions.to_hdf(data_dir + 'truncated_gbc_independent_predictions.h5', 'df')