__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from bck_stats.sklearn_estimator_suite import GbcAutoNtrees, ClassificationSuite
import os
import multiprocessing
import cPickle
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'

njobs = multiprocessing.cpu_count() - 1


class LastObservedValue(BaseEstimator):
    """An estimator assigning probability of 1.0 to the last observed class.
    """
    def fit(self, X, y):
        self.nclasses = len(np.unique(y))

    def predict(self, X):
        if self.nclasses == 2:
            # need to return log-odds ratio for binary classification
            y = np.zeros(X.shape[0], dtype=np.float64)
            last_value = X[:, -1]
            y[last_value == 1] = np.log(1000.0)
            y[last_value == 0] = np.log(1.0 / 1000.0)
            y = y.reshape(X.shape[0], 1)
        else:
            y = np.zeros((X.shape[0], self.nclasses), dtype=np.float64)
            # assign probability of 1.0 to last observed class
            idx = (np.asarray(range(X.shape[0])), X[:, -1].astype(np.int))
            y[idx] = 1.0
        return y


def logit(x):
    return np.log(x / (1.0 - x))


def transform_inputs(training_set, test_set):
    print 'Transforming inputs...'
    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)
    xmean = np.mean(np.append(training_set['time'].values, test_set['time'].values))
    xsigma = np.std(np.append(training_set['time'].values, test_set['time'].values))
    # standardize
    training_set['time'] = (training_set['time'] - xmean) / xsigma
    test_set['time'] = (test_set['time'] - xmean) / xsigma

    lincols = ['car_age', 'duration_previous']
    for c in lincols:
        # do sqrt transform since these predictors can have a value of zero
        training_set[c] = np.sqrt(training_set[c])
        test_set[c] = np.sqrt(test_set[c])
        # standardize
        xmean = np.mean(np.append(training_set[c].values, test_set[c].values))
        xsigma = np.std(np.append(training_set[c].values, test_set[c].values))
        # standardize
        training_set[c] = (training_set[c] - xmean) / xsigma
        test_set[c] = (test_set[c] - xmean) / xsigma

    logcols = ['car_value', 'risk_factor', 'age_oldest', 'age_youngest', 'cost']
    # first recenter values
    training_set['age_oldest'] -= 17.0
    training_set['age_youngest'] -= 15.0
    test_set['age_oldest'] -= 17.0
    test_set['age_youngest'] -= 15.0
    training_set['cost'] -= 250.0
    test_set['cost'] -= 250.0
    for c in logcols:
        # do log transform since these predictors can have a value of zero
        training_set[c] = np.log(training_set[c]).astype(np.float32)
        test_set[c] = np.log(test_set[c]).astype(np.float32)
        # standardize
        xmean = np.mean(np.append(training_set[c].values, test_set[c].values))
        xsigma = np.std(np.append(training_set[c].values, test_set[c].values))
        # standardize
        training_set[c] = (training_set[c] - xmean) / xsigma
        test_set[c] = (test_set[c] - xmean) / xsigma

    logit_cols = []
    for c in training_set.columns:
        if 'fraction' in c:
            logit_cols.append(c)

    for c in logit_cols:
        # do logit transform since these predictors can have a value of zero
        training_set[c][training_set[c] == 0] = 0.01
        training_set[c][training_set[c] == 1] = 0.99
        test_set[c][test_set[c] == 0] = 0.01
        test_set[c][test_set[c] == 1] = 0.99
        training_set[c] = logit(training_set[c])
        test_set[c] = logit(test_set[c])
        # standardize
        xmean = np.mean(np.append(training_set[c].values, test_set[c].values))
        xsigma = np.std(np.append(training_set[c].values, test_set[c].values))
        # standardize
        training_set[c] = (training_set[c] - xmean) / xsigma
        test_set[c] = (test_set[c] - xmean) / xsigma

    return training_set, test_set


def fit_fixed_shopping_pt(training_set, response, category, test_set):
    """
    Fit a suite of classifiers for each shopping point independently.

    :param training_set: Pandas DataFrame containing the training set features.
    :param response: The classes for this category, a Pandas Series object.
    """
    # refine the more computationally-demanding estimators less
    n_refinements = {'DecisionTreeClassifier': 2,
                     'RandomForestClassifier': 0,
                     'GbcAutoNtrees': 0}

    shopping_points = test_set.index.get_level_values(1).unique()
    test_cust_ids = test_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), len(n_refinements)), dtype=np.int), index=test_cust_ids,
                             columns=n_refinements.keys())
    y_predict['Combined'] = 0
    y_predict.name = category

    # add last plan value to end of predictors
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    train_last_value = training_set[categories]
    test_last_value = test_set[categories]

    training_set = training_set.drop(categories, axis=1)
    test_set = test_set.drop(categories, axis=1)

    for spt in range(2, max(shopping_points) + 1):
        print 'Training for shopping Point', spt
        X_train = training_set.xs(spt, level=1)
        X_test = test_set.xs(spt, level=1)
        y = response.ix[X_train.index]
        print 'Found', len(y), 'customers.'
        # transform and standardize predictors
        X_train, X_test = transform_inputs(X_train, X_test)

        # add last observed value to end, since this forms the base estimator for GBC
        this_train_last_value = train_last_value.xs(spt, level=1)[category].values
        if category in ['C', 'D', 'G']:
            # indicators for this category start at 1
            this_train_last_value -= 1
        X_train = np.append(X_train.values, this_train_last_value[:, np.newaxis], axis=1)

        # initialize the list of sklearn objects corresponding to different statistical models
        models = [DecisionTreeClassifier(),
                  RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=njobs, max_depth=30),
                  GbcAutoNtrees(subsample=0.5, n_estimators=1000, learning_rate=0.01, init=LastObservedValue())]
        # set the tuning ranges manually
        tuning_ranges = {'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
                         'RandomForestClassifier': {'max_features':
                                                    list(np.unique(np.logspace(np.log10(2),
                                                                               np.log10(X_train.shape[1] - 1),
                                                                               5).astype(np.int)))},
                         'GbcAutoNtrees': {'max_depth': [1, 2, 3]}}

        suite = ClassificationSuite(n_features=X_train.shape[1], tuning_ranges=tuning_ranges, njobs=njobs,
                                    cv=np.max((5, njobs)), verbose=True, models=models)
        suite.fit(X_train, y.values, n_refinements=n_refinements)

        print 'Pickling models...'
        cPickle.dump(suite, open(data_dir + 'models/sklearn_suite_shopping_point' + str(spt) + '_category' +
                                 category + '.pickle', 'wb'))

        print 'Validation accuracy scores:'
        for model_name in suite.best_scores:
            print model_name, suite.best_scores[model_name]

        # plot the validation error for each model
        fig, ax1 = plt.subplots()
        plt.bar(np.arange(0, len(suite.best_scores.keys())), suite.best_scores.values())
        plt.ylim(ymin=min(0.5, np.min(suite.best_scores.values())))
        xtickNames = plt.setp(ax1, xticklabels=suite.best_scores.keys())
        plt.setp(xtickNames, rotation=45)
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Model')
        plt.title(category + ', Fixed Shopping Point ' + str(spt))
        plt.tight_layout()
        plt.savefig(base_dir + 'plots/cverror_sklearn_suite_shopping_point' + str(spt) + '_category' + category +
                    '.png')

        # predict the test data
        print 'Predicting the class for the test set...'
        test_cust_ids = X_test.index  # only update customers that have made it to this shopping point
        this_test_last_value = test_last_value.xs(spt, level=1)[category].values
        if category in ['C', 'D', 'G']:
            # indicators for this category start at 1
            this_test_last_value -= 1
        X_test = np.append(X_test.values, this_test_last_value[:, np.newaxis], axis=1)
        spt_predict = suite.predict_all(X_test)
        for c in n_refinements.keys():  # loop over the models
            y_predict.ix[test_cust_ids, c] = spt_predict[c]
        y_predict.ix[test_cust_ids, 'Combined'] = suite.predict(X_test)

    return y_predict


def fit_random_shopping_pt(training_set, y, test_set):
    pass


if __name__ == "__main__":

    training_set = pd.read_hdf(data_dir + 'training_set.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set.h5', 'df')

    # for testing, reduce number of customers
    # customer_ids = training_set.index.get_level_values(0).unique()
    # training_set = training_set.select(lambda x: x[0] < customer_ids[1000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    shopping_points = training_set.index.get_level_values(1).unique()

    response_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    response = training_set[training_set['record_type'] == 1][response_columns]
    response = response.reset_index(level=1).drop(training_set.index.names[1], axis=1)

    training_set = training_set[training_set['record_type'] == 0]

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID',
                      'most_common_plan']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    predictions = []
    for category in response_columns:
        print 'Training category', category, '...'
        tstart = time.clock()
        this_prediction = fit_fixed_shopping_pt(training_set, response[category], category, test_set)
        predictions.append(this_prediction)
        tend = time.clock()
        print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'

    predictions = pd.concat(predictions, axis=1, keys=response_columns)

    predictions.to_hdf(data_dir + 'sklearn_suite_independent_predictions_fixedspt.h5', 'df')