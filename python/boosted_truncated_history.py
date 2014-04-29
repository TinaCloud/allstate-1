__author__ = 'brandonkelly'

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import VerboseReporter
from sklearn.utils import check_arrays, check_random_state, column_or_1d, array2d
from sklearn.ensemble.gradient_boosting import PriorProbabilityEstimator
from sklearn.ensemble._gradient_boosting import _random_sample_mask, predict_stages, predict_stage
from sklearn.tree._tree import DTYPE, PresortBestSplitter, FriedmanMSE
from sklearn.cross_validation import train_test_split
from sklearn.metrics import zero_one_loss
from truncated_tree_ensemble import get_truncated_shopping_indices
import time
import pandas as pd
import os
import cPickle
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'


class UniquePriorProbability(PriorProbabilityEstimator):

    def fit(self, X, y):
        classes, y = np.unique(y, return_inverse=True)
        self.classes = classes
        super(UniquePriorProbability, self).fit(X, y)


class LastObservedValuePartitioned(BaseEstimator):
    """An estimator assigning prior probabilities based on the last observed class.
    """
    def __init__(self, train_set, purchased_plan, p_last):
        p_last = p_last / p_last.sum()  # make sure probabilities sum to one
        classes, indices, y = np.unique(purchased_plan.values, return_index=True, return_inverse=True)
        old_to_new = dict()
        for k in range(len(classes)):
            # create inverse mapping that returns new class label given the old class label
            old_to_new[str(np.asscalar(purchased_plan.values[indices[k]]))] = k
        self.old_to_new = old_to_new
        self.nclasses_purchased = len(classes)
        self.nclasses = np.max(purchased_plan.values) + 1  # plan labels start at zero
        self.nclasses = 2304
        self.classes = classes
        self.priors = np.zeros((self.nclasses_purchased, self.nclasses))
        new_id = pd.Series(data=y, index=purchased_plan.index)
        for i in range(len(p_last)):
            # average over the shopping points
            spnt = i + 2
            last_plan = train_set.xs(spnt, level=1)['planID']
            for j in xrange(self.nclasses):
                class_counts = np.bincount(new_id.ix[last_plan[last_plan == j].index], minlength=len(classes))
                # priors[k, j] is fraction in class k (new label) with last observed value as class j (old label)
                if np.sum(class_counts) > 0:
                    self.priors[:, j] += p_last[i] * class_counts / float(np.sum(class_counts))

        # make sure there is always non-zero probability in going from last observed plan -> purchased plan
        prior_last_obs = np.zeros(self.nclasses_purchased)
        for j in range(self.nclasses_purchased):
            olabel = classes[j]
            nlabel = self.old_to_new[str(olabel)]
            prior_last_obs[j] = self.priors[nlabel, olabel]

        prior_last_obs[prior_last_obs == 0] = prior_last_obs[prior_last_obs > 0].mean()

        for j in range(self.nclasses_purchased):
            olabel = classes[j]
            nlabel = self.old_to_new[str(olabel)]
            self.priors[nlabel, olabel] = prior_last_obs[j]

        prior_norm = self.priors.sum(axis=0)
        prior_norm[prior_norm == 0] = 1.0  # don't divide by zero
        self.priors /= prior_norm  # normalize so probabilities sum to one

    def fit(self, X, y):
        return self

    def predict(self, last_obs_plan):
        y_pred = np.zeros((len(last_obs_plan), self.nclasses_purchased))
        uplans = np.unique(last_obs_plan)  # last observed plan uses old labeling
        for up in uplans:
            idx = np.where(last_obs_plan == up)[0]
            y_pred[idx] = self.priors[:, up]

        return y_pred


class LastObservedValue(BaseEstimator):
    """An estimator assigning prior probability of one based on the last observed class.
    """
    def __init__(self, purchased_plan):
        classes, indices, y = np.unique(purchased_plan.values, return_index=True, return_inverse=True)
        old_to_new = dict()
        for k in range(len(classes)):
            # create inverse mapping that returns new class label given the old class label
            old_to_new[str(purchased_plan.values[indices[k]])] = k
        self.old_to_new = old_to_new
        self.nclasses = len(classes)
        self.classes = classes

    def fit(self, X, y):
        return self

    def predict(self, last_obs_plan):
        # assign probability one to last observed value
        y_pred = np.zeros((len(last_obs_plan), self.nclasses), dtype=np.float64)
        uplans = np.unique(last_obs_plan)
        for up in uplans:
            idx = np.where(last_obs_plan == up)[0]
            y_idx = self.old_to_new[up]
            y_pred[idx, y_idx] = 1.0

        return y_pred


class TruncatedHistoryGBC(GradientBoostingClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=0.1,
                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None,
                 verbose=0, fix_history=False):
        super(TruncatedHistoryGBC, self).__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                                  subsample=subsample, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_depth=max_depth, init=init,
                                                  random_state=random_state, max_features=max_features, verbose=verbose)
        self.classes_ = None
        self.n_classes_ = 0
        self.test_score = np.zeros(self.n_estimators)
        self.fix_history = fix_history

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
            if not self.fix_history:
                trunc_idx = get_truncated_shopping_indices(n_shop)
                X_trunc = X[trunc_idx]
            else:
                X_trunc = X

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

    def fit_all(self, X, y, n_shop, last_obs_plan):
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
            if self.verbose:
                print 'Initializing gradient boosting...'
            # init state
            self._init_state()

            # fit initial model
            if not self.fix_history:
                idx = get_truncated_shopping_indices(n_shop)
            else:
                idx = np.arange(len(n_shop))

            # init predictions by averaging over the shopping histories
            y_pred = self.init_.predict(last_obs_plan[idx])
            print 'First training accuracy:', accuracy_score(y, y_pred.argmax(axis=1))
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

    def fit(self, X, y, n_shop=None, last_obs_plan=None, X_test=None, y_test=None, last_plan_test=None,
            n_shop_test=None, fix_ntrees=False):
        try:
            n_shop is not None
        except ValueError:
            "Must supply n_shop."
        try:
            last_obs_plan is not None
        except ValueError:
            "Must supply last_obs_plan"

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # initial pass through the data
        self.fit_all(X, y, n_shop, last_obs_plan)

        if not fix_ntrees:
            # find where the validation score is maximized, only use this many trees
            print 'Getting test error...'

            initial_test_score = 0.0
            if not self.fix_history:
                n_histories = 50
                for h in range(n_histories):
                    idx = get_truncated_shopping_indices(n_shop_test)
                    y_predict = self.classes_[self.init_.predict(last_plan_test[idx]).argmax(axis=1)]
                    initial_test_score_h = accuracy_score(y_test, y_predict)
                    initial_test_score += initial_test_score_h
                initial_test_score /= n_histories
            else:
                y_predict = self.classes_[self.init_.predict(last_plan_test).argmax(axis=1)]
                initial_test_score = accuracy_score(y_test, y_predict)

            test_score = np.zeros(self.n_estimators)
            train_score = np.zeros(self.n_estimators)
            if not self.fix_history:
                n_histories = 50
            else:
                n_histories = 1
            for h in xrange(n_histories):
                # test accuracy score
                if not self.fix_history:
                    idx = get_truncated_shopping_indices(n_shop_test)
                else:
                    idx = np.arange(len(n_shop_test))
                y_predict = self.staged_predict(X_test[idx], last_plan_test[idx])
                for i, yp in enumerate(y_predict):
                    test_score[i] += accuracy_score(y_test, yp)
                # train accuracy score
                if not self.fix_history:
                    idx = get_truncated_shopping_indices(n_shop)
                else:
                    idx = np.arange(len(n_shop))
                y_predict = self.staged_predict(X[idx], last_obs_plan[idx])
                for i, yp in enumerate(y_predict):
                    train_score[i] += accuracy_score(self.classes_[y], yp)
            test_score /= n_histories
            train_score /= n_histories

            print 'Validation error using last observed value:', initial_test_score

            # ntrees = test_score.argmax() + 1
            # if self.verbose:
            #     print 'Chose', ntrees, 'based on the validation score.'
            # self.n_estimators = ntrees
            # self.estimators_ = self.estimators_[:ntrees]

            plt.plot(self.train_score_)
            plt.ylabel('Training deviance loss')
            plt.show()

            plt.plot(test_score, label='validation', lw=2)
            plt.plot(train_score, label='train', lw=2)
            plt.ylabel('Accuracy')
            plt.xlabel('Number of Trees')
            plt.plot(plt.xlim(), (initial_test_score, initial_test_score), '-.')
            plt.legend(loc='best')
            plt.savefig(base_dir + 'plots/truncated_gbc_validation_error_maxdepth' + str(self.max_depth) + '.png')
            plt.show()

        return self

    def _init_decision_function(self, last_obs_plan):
        score = self.init_.predict(last_obs_plan).astype(np.float64)
        return score

    def decision_function(self, X, last_obs_plan=None):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        try:
            last_obs_plan is not None
        except ValueError:
            "must supply last_obs_plan"
        X = array2d(X, dtype=DTYPE, order="C")
        score = self._init_decision_function(last_obs_plan)
        predict_stages(self.estimators_, X, self.learning_rate, score)
        return score

    def predict_proba(self, X, last_obs_plan=None):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        try:
            last_obs_plan is not None
        except ValueError:
            "must supply last_obs_plan"
        score = self.decision_function(X, last_obs_plan)
        return self._score_to_proba(score)

    def predict(self, X, last_obs_plan=None):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        try:
            last_obs_plan is not None
        except ValueError:
            "must supply last_obs_plan"
        proba = self.predict_proba(X, last_obs_plan)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def staged_decision_function(self, X, last_obs_plan=None):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = array2d(X, dtype=DTYPE, order="C")
        score = self._init_decision_function(last_obs_plan)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, score)
            yield score

    def staged_predict_proba(self, X, last_obs_plan=None):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for score in self.staged_decision_function(X, last_obs_plan):
            yield self._score_to_proba(score)

    def staged_predict(self, X, last_obs_plan=None):
        """Predict classes at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for proba in self.staged_predict_proba(X, last_obs_plan):
            yield self.classes_.take(np.argmax(proba, axis=1), axis=0)


def validate_gbc_tree(args):
    X_train, y_train, X_val, y_val, n_shop, grid, last_obs_idx = args

    cv_score = np.zeros(len(grid))

    print 'Calculating CV scores for tuning parameters:'
    for j, params in enumerate(grid):
        print j, params
        trunc_tree = TruncatedHistoryGBC(subsample=0.9, learning_rate=0.01, n_estimators=ntrees, max_depth=params,
                                         verbose=True)
        trunc_tree.fit(X_train, y_train, n_shop)
        y_predict = trunc_tree.predict(X_val)
        cv_score[j] = accuracy_score(y_val, y_predict)

    return cv_score


def fit_and_predict_test_plans(training_set, response, test_set, last_test_plan, ntrees, n_shop, max_depth):

    test_cust_ids = test_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), 2), dtype=np.int), index=test_cust_ids,
                             columns=['TruncatedGBC', 'LastObsValue'])

    # need to get indices for last observed value
    n_shop_test = np.asarray([test_set.ix[cid].index[-1] for cid in test_cust_ids])

    # get prior fraction truncated at each shopping point
    shopping_counts = np.bincount(n_shop_test, minlength=1+n_shop.max())
    p_last = shopping_counts[2:] / float(np.sum(shopping_counts[2:]))

    # do traint/test split to monitor test error as a function of number of trees
    customer_ids = training_set.index.get_level_values(0).unique()
    customer_idx = np.arange(0, len(customer_ids))
    customers_train, customers_test = train_test_split(customer_idx)
    customers_train = np.sort(customers_train)
    customers_test = np.sort(customers_test)

    print 'Doing train/validation split...'
    validation_set = training_set.reset_index(level=1).ix[customer_ids[customers_test]]
    validation_set = validation_set.drop('planID', axis=1)
    validation_response = response.ix[customer_ids[customers_test]]
    validation_plans = training_set.reset_index(level=1).ix[customer_ids[customers_test]]['planID']
    validation_n_shop = n_shop.ix[customer_ids[customers_test]]

    train_idx = []
    for train_customer in customer_ids[customers_train]:
        this_train = training_set.ix[train_customer]
        train_idx.extend([(train_customer, spt) for spt in this_train.index])

    train_set = training_set.ix[train_idx]
    train_response = response.ix[customer_ids[customers_train]]
    train_plans = train_set.reset_index(level=1)['planID']
    train_n_shop = n_shop.ix[customer_ids[customers_train]]

    print 'Initializing estimator...'

    init = LastObservedValuePartitioned(train_set, train_response, p_last)
    # init = UniquePriorProbability()
    init.fit(train_set.values, np.squeeze(train_response.values))
    train_set = train_set.reset_index(level=1).drop('planID', axis=1)

    print 'Using features', train_set.columns

    subsample = 0.9
    lrate = 0.01

    trunc_tree = TruncatedHistoryGBC(subsample=subsample, learning_rate=lrate, n_estimators=ntrees, max_depth=max_depth,
                                     verbose=True, init=init, fix_history=True)

    train_idx = get_truncated_shopping_indices(train_n_shop.values)
    val_idx = get_truncated_shopping_indices(validation_n_shop.values)

    trunc_tree.fit(train_set.values[train_idx], np.squeeze(train_response.values), train_n_shop.values,
                   train_plans.values[train_idx], validation_set.values[val_idx],
                   np.squeeze(validation_response.values),
                   validation_plans.values[val_idx], validation_n_shop.values)

    # trunc_tree = GradientBoostingClassifier(learning_rate=0.01, n_estimators=ntrees, max_depth=max_depth, verbose=1)
    idx = get_truncated_shopping_indices(train_n_shop.values)
    # trunc_tree.fit(train_set.values[idx], np.squeeze(train_response.values))

    train_predict_init = trunc_tree.init_.predict(train_plans.values[idx]).argmax(axis=1)
    train_predict_init = trunc_tree.init_.classes[train_predict_init]
    print 'Initial train accuracy:', accuracy_score(np.squeeze(train_response.values), train_predict_init)
    # idx = get_truncated_shopping_indices(train_n_shop.values)
    train_predict = trunc_tree.predict(train_set.values[idx], train_plans.values[idx])
    # train_predict = trunc_tree.predict(train_set.values[idx])
    idx = get_truncated_shopping_indices(validation_n_shop.values)
    val_predict_init = trunc_tree.init_.predict(validation_plans.values[idx]).argmax(axis=1)
    val_predict_init = trunc_tree.init_.classes[val_predict_init]
    print 'Initial validation accuracy:', accuracy_score(np.squeeze(validation_response.values), val_predict_init)
    test_predict = trunc_tree.predict(validation_set.values[idx], validation_plans.values[idx])
    # test_predict = trunc_tree.predict(validation_set.values[idx])
    print 'Training accuracy:', accuracy_score(np.squeeze(train_response.values), train_predict)
    print 'Validation accuracy:', accuracy_score(np.squeeze(validation_response.values), test_predict)

    ntrees = trunc_tree.n_estimators
    del trunc_tree

    print 'Resetting Initializer...'
    init = LastObservedValuePartitioned(training_set, response, p_last)
    # include the shopping point as a predictor and remove the planID from the feature set.
    # need to do this after the constructor for LastObservedValuePartitioned
    training_set = training_set.reset_index(level=1).drop('planID', axis=1)

    print 'Refitting using all the data...'
    trunc_tree = TruncatedHistoryGBC(subsample=subsample, learning_rate=lrate, n_estimators=ntrees, max_depth=max_depth,
                                     verbose=True, init=init)
    trunc_tree.fit(training_set.values, np.squeeze(response.values), n_shop.values, training_set['planID'].values,
                   validation_set.values, np.squeeze(validation_response.values), validation_plans.values,
                   validation_n_shop.values, fix_ntrees=True)

    # predict the test data
    print 'Predicting the class for the test set...'
    first_spt_idx = np.roll(np.cumsum(n_shop_test), 1)
    first_spt_idx[0] = 0
    # need the minus one here since the shopping points index starts at one and n_trunc starts at 2
    last_idx = first_spt_idx + n_shop_test - 1
    test_set = test_set.reset_index(level=1)
    test_set = test_set.iloc[last_idx]

    y_predict['TruncatedGBC'] = last_test_plan
    # need to find those customers in the test set with last observed plan among those plans in the training set
    classes = init.classes
    test_in_train = last_test_plan.apply(lambda x: x in classes)
    test_set = test_set[test_in_train]
    # override last observed plan for those customers without last observed plan in the training set
    test_set = test_set.drop('planID', axis=1)
    y_predict.ix[test_set.index, 'TruncatedGBC'] = trunc_tree.predict(test_set.values, last_test_plan)
    y_predict['LastObsValue'] = last_test_plan

    return y_predict, trunc_tree


if __name__ == "__main__":

    ntrees = 20
    max_depth = 1

    training_set = pd.read_hdf(data_dir + 'training_set_v2.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set_v2.h5', 'df')

    response = training_set[training_set['record_type'] == 1]['planID']
    response = response.reset_index(level=1)['planID']

    plan_counts = np.bincount(response)
    top_plans = np.where(plan_counts > 50)[0]

    print 'Found', len(top_plans), 'plans.'

    response = response[response.apply(lambda x: x in top_plans)]
    kept_customers = response.index
    kept_idx = []
    for kept in kept_customers:
        this_df = training_set.ix[kept]
        kept_idx.extend([(kept, spt) for spt in this_df.index])

    training_set = training_set.ix[kept_idx]

    # for testing, reduce number of customers
    customer_ids = training_set.index.get_level_values(0).unique()
    training_set = training_set.select(lambda x: x[0] < customer_ids[5000], axis=0)
    customer_ids = test_set.index.get_level_values(0).unique()
    test_set = test_set.select(lambda x: x[0] < customer_ids[5000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    last_spt_idx = [(cid, test_set.ix[cid].index[-1]) for cid in test_set.index.get_level_values(0).unique()]
    last_observed_value = test_set.ix[last_spt_idx].reset_index(level=1)['planID']

    training_set = training_set[training_set['record_type'] == 0]
    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])
    n_shop = pd.Series(n_shop, index=customer_ids)

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'most_common_plan', 'A',
                      'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    tstart = time.clock()
    prediction, trunc_gbc = fit_and_predict_test_plans(training_set, response, test_set, last_observed_value, ntrees,
                                                       n_shop, max_depth)
    tend = time.clock()
    print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

    # helpful to compare with as a sanity check
    prediction['LastObsValue'] = last_observed_value

    print 'Pickling models...'
    cPickle.dump(trunc_gbc, open(data_dir + 'models/truncated_gbc_max_depth' + str(max_depth) + '.pickle', 'wb'))
    prediction.to_hdf(data_dir + 'truncated_gbc_predictions_max_depth' + str(max_depth) + '.h5', 'df')

    # feature importance plots
    fimportance = np.array(trunc_gbc.feature_importances_)
    fimportance /= fimportance.max()
    features = np.array(training_set.reset_index(level=1).drop('planID'))
    sorted_idx = np.argsort(fimportance)[::-1]
    for i in xrange(len(fimportance)):
        print i, features[sorted_idx[i]], fimportance[sorted_idx[i]]

    pos = np.arange(30) + 0.5
    plt.barh(pos, fimportance[sorted_idx], align='center')
    plt.yticks(pos, features[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Truncated GBC, max_depth = ' + str(max_depth))
    plt.savefig(base_dir + 'plots/truncated_gbc_feature_importance_max_depth' + str(max_depth) + '.png')
    plt.show()
