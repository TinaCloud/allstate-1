__author__ = 'brandonkelly'

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
    def __init__(self, train_plans, purchased_plan):
        classes, indices, y = np.unique(purchased_plan.values, return_index=True, return_inverse=True)
        lov_classes, lov_indices, y_lov = np.unique(train_plans.values, return_index=True, return_inverse=True)
        old_to_new_purchased = dict()
        old_to_new_lov = dict()
        for k in range(len(classes)):
            # create inverse mapping that returns new class label given the old class label
            old_to_new_purchased[str(np.asscalar(purchased_plan.values[indices[k]]))] = k
        for k in range(len(lov_classes)):
            old_to_new_lov[str(np.asscalar(train_plans.values[lov_indices[k]]))] = k
        self.old_to_new = old_to_new_purchased
        self.old_to_new_lov = old_to_new_lov
        self.nclasses_purchased = len(classes)
        self.nclasses_lov = len(np.unique(train_plans.values))
        self.classes = classes
        self.classes_lov = lov_classes
        self.priors = np.zeros((self.nclasses_purchased, self.nclasses_lov))
        new_id = pd.Series(data=y, index=purchased_plan.index)
        for j in xrange(self.nclasses_lov):
            class_counts = np.bincount(new_id.ix[train_plans[train_plans == lov_classes[j]].index],
                                       minlength=len(classes))
            # priors[k, j] is fraction in class k (new label) with last observed value as class j (new label)
            if np.sum(class_counts) > 0:
                self.priors[:, j] = class_counts / float(np.sum(class_counts))

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
            y_pred[idx] = self.priors[:, self.old_to_new_lov[str(up)]]

        return y_pred


class TruncatedHistoryGBC(GradientBoostingClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=0.1,
                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None,
                 verbose=0):
        super(TruncatedHistoryGBC, self).__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                                  subsample=subsample, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_depth=max_depth, init=init,
                                                  random_state=random_state, max_features=max_features, verbose=verbose)
        self.classes_ = None
        self.n_classes_ = 0
        self.test_score = np.zeros(self.n_estimators)

    def fit_all(self, X, y, last_obs_plan):
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

            # init predictions
            y_pred = self.init_.predict(last_obs_plan)
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
        n_stages = self._fit_stages(X, y, y_pred, random_state, begin_at_stage)
        # change shape of arrays after fit (early-stopping or additional tests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
            if hasattr(self, '_oob_score_'):
                self._oob_score_ = self._oob_score_[:n_stages]

        return self

    def fit(self, X, y, last_obs_plan=None, X_test=None, y_test=None, last_plan_test=None, fix_ntrees=False):
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
        self.fit_all(X, y, last_obs_plan)

        if not fix_ntrees:
            # find where the validation score is maximized, only use this many trees
            print 'Getting test error...'

            y_predict = self.classes_[self.init_.predict(last_plan_test).argmax(axis=1)]
            initial_test_score = accuracy_score(y_test, y_predict)
            y_predict = self.classes_[self.init_.predict(last_obs_plan).argmax(axis=1)]
            initial_train_score = accuracy_score(self.classes_[y], y_predict)

            test_score = np.zeros(self.n_estimators)
            train_score = np.zeros(self.n_estimators)
            # test accuracy score
            y_predict = self.staged_predict(X_test, last_plan_test)
            for i, yp in enumerate(y_predict):
                test_score[i] = accuracy_score(y_test, yp)
            # train accuracy score
            y_predict = self.staged_predict(X, last_obs_plan)
            for i, yp in enumerate(y_predict):
                train_score[i] = accuracy_score(self.classes_[y], yp)

            print 'Validation accuracy using last observed value:', initial_test_score
            print 'Train accuracy using last observed value:', initial_train_score

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


def fit_and_predict_test_plans(training_set, response, test_set, ntrees, max_depth):

    test_cust_ids = test_set.index

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), 2), dtype=np.int), index=test_cust_ids,
                             columns=['TruncatedGBC', 'LastObsValue'])

    # do train/test split to monitor test error as a function of number of trees
    customer_ids = training_set.index.get_level_values(0).unique()
    customer_idx = np.arange(0, len(customer_ids))
    customers_train, customers_test = train_test_split(customer_idx)
    customers_train = np.sort(customers_train)
    customers_test = np.sort(customers_test)

    print 'Doing train/validation split...'
    validation_set = training_set.ix[customer_ids[customers_test]]
    validation_set = validation_set.drop('planID', axis=1)
    validation_response = response.ix[customer_ids[customers_test]]
    validation_plans = training_set.ix[customer_ids[customers_test]]['planID']

    train_set = training_set.ix[customer_ids[customers_train]]
    train_response = response.ix[customer_ids[customers_train]]
    train_plans = train_set['planID']

    print 'Initializing estimator...'

    init = LastObservedValuePartitioned(train_plans, train_response)
    # init = UniquePriorProbability()
    # init.fit(train_set.values, np.squeeze(train_response.values))

    train_set = train_set.drop('planID', axis=1)

    print 'Using features', train_set.columns

    train_predict_init = init.predict(train_plans.values).argmax(axis=1)
    train_predict_init = init.classes[train_predict_init]
    print 'Initial train accuracy:', accuracy_score(np.squeeze(train_response.values), train_predict_init)
    val_predict_init = init.predict(validation_plans.values).argmax(axis=1)
    val_predict_init = init.classes[val_predict_init]
    print 'Initial validation accuracy:', accuracy_score(np.squeeze(validation_response.values), val_predict_init)

    subsample = 0.9
    lrate = 0.01

    trunc_tree = TruncatedHistoryGBC(subsample=subsample, learning_rate=lrate, n_estimators=ntrees, max_depth=max_depth,
                                     verbose=True, init=init)

    trunc_tree.fit(train_set.values, train_response.values, train_plans.values, validation_set.values,
                   validation_response.values, validation_plans.values)

    # trunc_tree = GradientBoostingClassifier(learning_rate=0.01, n_estimators=ntrees, max_depth=max_depth, verbose=1)
    # trunc_tree.fit(train_set.values, train_response.values)

    train_predict = trunc_tree.predict(train_set.values, train_plans.values)
    # train_predict = trunc_tree.predict(train_set.values)
    test_predict = trunc_tree.predict(validation_set.values, validation_plans.values)
    # test_predict = trunc_tree.predict(validation_set.values)
    print 'Training accuracy:', accuracy_score(np.squeeze(train_response.values), train_predict)
    print 'Validation accuracy:', accuracy_score(np.squeeze(validation_response.values), test_predict)

    ntrees = trunc_tree.n_estimators
    del trunc_tree

    print 'Resetting Initializer...'
    init = LastObservedValuePartitioned(training_set, response)
    # include the shopping point as a predictor and remove the planID from the feature set.
    # need to do this after the constructor for LastObservedValuePartitioned
    training_set = training_set.drop('planID', axis=1)

    print 'Refitting using all the data...'
    trunc_tree = TruncatedHistoryGBC(subsample=subsample, learning_rate=lrate, n_estimators=ntrees, max_depth=max_depth,
                                     verbose=True, init=init)
    trunc_tree.fit(training_set.values, response.values, training_set['planID'].values, validation_set.values,
                   validation_response.values, validation_plans.values, fix_ntrees=True)

    # predict the test data
    last_test_plan = test_set['planID']
    train_plans = training_set['planID']
    predict_customers = last_test_plan.apply(lambda x: x in train_plans)

    print 'Predicting the class for the test set...'
    print 'Found', np.sum(predict_customers), 'customers in the test set that we can make GBC predictions for.'

    y_predict['TruncatedGBC'] = last_test_plan
    test_set = test_set.drop('planID', axis=1)
    # predict the classes when we have enough training data
    y_predict.ix[predict_customers.values, 'TruncatedGBC'] = trunc_tree.predict(test_set.values, last_test_plan)
    y_predict['LastObsValue'] = last_test_plan

    return y_predict, trunc_tree


if __name__ == "__main__":

    ntrees = 1000
    max_depth = 2

    training_set = pd.read_hdf(data_dir + 'training_set_v2.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set_v2.h5', 'df')

    # for testing, reduce number of customers
    customer_ids = training_set.index.get_level_values(0).unique()
    training_set = training_set.select(lambda x: x[0] < customer_ids[30000], axis=0)
    customer_ids = test_set.index.get_level_values(0).unique()
    test_set = test_set.select(lambda x: x[0] < customer_ids[30000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    response = training_set[training_set['record_type'] == 1]['planID']
    response = response.reset_index(level=1)['planID']

    training_set = training_set[training_set['record_type'] == 0]

    # only train on plans that have at least 50 training points. just use last observed value for remainders
    plan_counts = np.bincount(response)
    top_plans = np.where(plan_counts > 25)[0]

    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])
    n_shop = pd.Series(n_shop, index=customer_ids)

    idx = get_truncated_shopping_indices(n_shop.values)
    truncated_spt = training_set.reset_index(level=1)['shopping_pt'].values[idx]

    # only keep values from truncated history
    midx = zip(customer_ids, truncated_spt)
    training_set = training_set.ix[midx].reset_index(level=1)

    # only keep customer with common plans in the training set
    last_plans = training_set['planID']
    common = last_plans.apply(lambda x: x in top_plans)
    response = response.ix[common]
    training_set = training_set.ix[common]

    print 'Found', len(training_set['planID']), 'customers for GBC training.'
    print 'Found', len(response.unique()), 'plans purchased among this set.'

    # only keep last value in test set history
    test_idx = [(cid, test_set.ix[cid].index[-1]) for cid in test_set.index.get_level_values(0).unique()]
    test_set = test_set.ix[test_idx].reset_index(level=1)

    # drop useless predictors
    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'most_common_plan', 'A',
                      'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    tstart = time.clock()
    prediction, trunc_gbc = fit_and_predict_test_plans(training_set, response, test_set, ntrees, max_depth)
    tend = time.clock()
    print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

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
