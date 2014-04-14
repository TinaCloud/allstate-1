__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn_estimator_suite import GbcAutoNtrees, ClassificationSuite
import os
import multiprocessing
import cPickle
import time

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'

njobs = multiprocessing.cpu_count() - 1


def fit_fixed_shopping_pt(training_set, response, category, test_set):
    """
    Fit a suite of classifiers for each shopping point independently.

    :param training_set: Pandas DataFrame containing the training set features.
    :param response: The classes for this category, a Pandas Series object.
    """
    # refine the more computationally-demanding estimators less
    n_refinements = {'LogisticRegression': 3,
                     'DecisionTreeClassifier': 3,
                     'LinearSVC': 3,
                     'RandomForestClassifier': 1,
                     'GbcAutoNtrees': 1}

    shopping_points = training_set.index.get_level_values(1).unique()

    test_cust_ids = test_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), len(n_refinements)), dtype=np.int), index=test_cust_ids,
                             columns=n_refinements.keys())
    y_predict.name = category

    # for spt in range(2, max(shopping_points) + 1):
    for spt in range(2, 5):
        print 'Training for shopping Point', spt
        X = training_set.xs(spt, level=1)
        y = response.ix[X.index].values
        X = X.values

        suite = ClassificationSuite(n_features=X.shape[1], njobs=njobs, cv=np.max((5, njobs)), verbose=True)
        suite.fit(X, y, n_refinements=n_refinements)

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
        X_test = test_set.xs(spt, level=1)
        test_cust_ids = X_test.index  # only update customers that have made it to this shopping point
        spt_predict = suite.predict_all(X_test.values)
        for c in y_predict.columns:  # loop over the models
            y_predict.ix[test_cust_ids, c] = spt_predict[c]

    return y_predict


def fit_random_shopping_pt(training_set, y, test_set):
    pass


if __name__ == "__main__":

    training_set = pd.read_hdf(data_dir + 'training_set.h5', 'df')
    test_set = pd.read_hdf(data_dir + 'test_set.h5', 'df')

    # fill_values = {'risk_factor': training_set['risk_factor'].mean(),
    #                'duration_previous': training_set['duration_previous'].mean()}
    #
    # training_set = training_set.fillna(value=fill_values)
    # test_set = test_set.fillna(value=fill_values)
    #
    # customer_ids = training_set.index.get_level_values(0).unique()

    # for testing, reduce number of customers
    # training_set = training_set.select(lambda x: x[0] < customer_ids[1000], axis=0)

    customer_ids = training_set.index.get_level_values(0).unique()

    shopping_points = training_set.index.get_level_values(1).unique()

    response_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    response = training_set[training_set['record_type'] == 1][response_columns]
    response = response.reset_index(level=1).drop(training_set.index.names[1], axis=1)

    training_set = training_set[training_set['record_type'] == 0]

    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID']
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