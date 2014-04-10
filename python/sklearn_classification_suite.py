__author__ = 'brandonkelly'

import numpy as np
import abc

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.cross_validation import KFold

float_types = [float, np.float, np.float32, np.float64, np.float_, np.float128, np.float16]
int_types = [int, np.int, np.int8, np.int16, np.int32, np.int64]


class BasePredictorSuite(object):
    """ Base class for running a pipeline using a set of algorithms from scikit-learn. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, tuning_ranges=None, models=None, standardize=True, cv=None, njobs=1, pre_dispatch='2*n_jobs',
                 blend=True, weights='score'):
        super(BasePredictorSuite, self).__init__()
        if tuning_ranges is None:
            tuning_ranges = dict()
        self.tuning_ranges = tuning_ranges
        if models is None:
            models = []
        self.models = models
        self.model_names = []
        for model in self.models:
            # store the names of the sklearn classes used
            self.model_names.append(model.__class__.__name__)
            try:
                # make sure the model names are in the dictionary of tuning parameters
                model.__class__.__name__ in tuning_ranges
            except ValueError:
                print 'Could not find tuning parameters for', model.__class__.__name__

        self.standardize = standardize
        if cv is None:
            self.cv = 3
        self.njobs = njobs
        self.pre_dispatch = pre_dispatch
        self.scorer = None
        self.blend = blend
        self.weights = weights
        self.best_scores = dict()

    def refine_grid(self, best_params, model_name):
        for param_name in best_params:
            pvalue_list = self.tuning_ranges[model_name][param_name]
            # find the values corresponding to
            idx = pvalue_list.index(param_name)
            ngrid = len(pvalue_list)
            if idx == 0:
                # first element of grid, so expand below it
                if type(pvalue_list[0]) in int_types:
                    pv_min = pvalue_list[0] / 2  # reduce minimum grid value by a factor of 2
                    pv_min = np.log10(max(1, pv_min))  # assume integer tuning parameters are never less than 1.
                    pv_max = np.log10(pvalue_list[1])
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.logspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    dp = np.log10(pvalue_list[1]) - np.log10(pvalue_list[0])
                    pv_min = np.log10(pvalue_list[0]) - dp
                    pv_max = np.log10(pvalue_list[1])
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))
            elif idx == ngrid - 1:
                # last element of grid, so expand above it
                if pvalue_list[idx] is None:
                    # special situation for some estimators, like the DecisionTreeClassifier
                    pv_min = pvalue_list[idx-1]  # increase the maximum grid value by a factor of 2
                    pv_max = np.log10(2 * pv_min)
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.logspace(pv_min, pv_max, ngrid-1).astype(np.int)))
                    # make sure we keep None as the last value in the list
                    self.tuning_ranges[model_name][param_name].append(None)
                elif type(pvalue_list[idx]) in int_types:
                    pv_min = pvalue_list[idx-1]  # increase the maximum grid value by a factor of 2
                    pv_max = np.log10(2 * pvalue_list[1])
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.logspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    dp = np.log10(pvalue_list[idx]) - np.log10(pvalue_list[idx-1])
                    pv_min = np.log10(pvalue_list[idx-1])
                    pv_max = np.log10(pvalue_list[idx]) + dp
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))
            else:
                # inner element of grid
                if pvalue_list[idx + 1] is None:
                    # special situation for some estimators, like the DecisionTreeClassifier
                    pv_min = pvalue_list[idx-1]  # increase the maximum grid value by a factor of 2
                    pv_max = np.log10(2 * pvalue_list[idx])
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.logspace(pv_min, pv_max, ngrid-1).astype(np.int)))
                    # make sure we keep None as the last value in the list
                    self.tuning_ranges[model_name][param_name].append(None)
                elif type(pvalue_list[idx]) in int_types:
                    pv_min = pvalue_list[idx-1]
                    pv_max = pvalue_list[idx+1]
                    # switch to linear spacing for interior integer grid values
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.linspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    pv_min = np.log10(pvalue_list[idx-1])
                    pv_max = np.log10(pvalue_list[idx+1])
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))

    def fit(self, X, y, n_refinements=1):
        ndata = len(y)
        try:
            X.shape[0] == ndata
        except ValueError:
            print 'X and y must have same number of rows.'

        if type(self.cv) == int:
            # construct cross-validation iterator
            self.cv = KFold(ndata, n_folds=self.cv)
        elif self.cv.n != ndata:
            # need to reconstruct cross-validation iterator since we have different data
            self.cv = KFold(ndata, n_folds=self.cv.n_folds)

        for k in range(len(self.models)):
            # fit tuning parameters for each model sequentially via cross-validation
            print 'Doing cross-validation for model', self.model_names[k], '...'
            grid = GridSearchCV(self.models[k], self.tuning_ranges[self.model_names[k]], scoring=self.scorer,
                                n_jobs=self.njobs, cv=self.cv, pre_dispatch=self.pre_dispatch)
            grid.fit(X, y)

            print 'Best', self.model_names[k], 'has:'
            for tuning_parameter in self.tuning_ranges[self.model_names[k]]:
                print '    ', tuning_parameter, '=', grid.best_params_[tuning_parameter]
            print '     CV Score of', grid.best_score_

            self.models[k] = grid.best_estimator_
            self.best_scores[self.model_names[k]] = grid.best_score_

            for i in range(n_refinements):
                old_score = grid.best_score_
                # now refine the grid and refit
                self.refine_grid(grid.best_params_, self.model_names[k])
                grid = GridSearchCV(self.models[k], self.tuning_ranges[self.model_names[k]], scoring=self.scorer,
                                    n_jobs=self.njobs, cv=self.cv, pre_dispatch=self.pre_dispatch)
                grid.fit(X, y)
                print 'Best', self.model_names[k], 'has:'
                for tuning_parameter in self.tuning_ranges[self.model_names[k]]:
                    print '    ', tuning_parameter, '=', grid.best_params_[tuning_parameter]
                print '     New CV Score of', grid.best_score_, 'is an improvement of', \
                    100.0 * (grid.best_score_ - old_score) / np.abs(old_score), '%.'

                self.models[k] = grid.best_estimator_
                self.best_scores[self.model_names[k]] = grid.best_score_

        return self

    def predict(self, X):
        y_predict_all = dict()
        for model, name in zip(self.models, self.model_names):
            y_predict_all[name] = model.predict(X)

        if self.blend:
            # average the model outputs
            y_predict = 0.0
            wsum = 0.0
            weight = 1.0
            for name in y_predict_all:
                if self.weights == 'score':
                    # weighted average using the CV scores for each model
                    weight = self.best_scores[name]
                y_predict += weight * y_predict_all[name]
                wsum += weight

            y_predict /= wsum

        else:
            y_predict = y_predict_all

        return y_predict


class ClassificationSuite(BasePredictorSuite):

    def __init__(self, tuning_ranges=None, models=None):
        if tuning_ranges is None:
            # use default values for grid search over tuning parameters for all models
            tuning_ranges = {'LogisticRegression': {'C': list(np.logspace(-3.0, 0.0, 5))},
                             'MultinomialNB': {'alpha': list(np.logspace(-3.0, 0.0, 5))},
                             'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
                             'LinearSVC': {'C': list(np.logspace(-3.0, 0.0, 5))},
                             'RandomForestClassifier': {'n_estimators': [20, 40, 80, 160, 320],
                                                        'max_features': [2, 4, 8, 16, 32]},
                             'GradientBoostingClassifier': {'max_depth': [1, 2, 3, 5, 10]}}
        if models is None:
            # initialize the list of sklearn objects corresponding to different statistical models
            models = []
            if 'LogisticRegression' in tuning_ranges:
                models.append(LogisticRegression(penalty='l1', class_weight='auto'))
            if 'MultinomialNB' in tuning_ranges:
                models.append(MultinomialNB())
            if 'DecisionTreeClassifier' in tuning_ranges:
                models.append(DecisionTreeClassifier())
            if 'LinearSVC' in tuning_ranges:
                models.append(LinearSVC(penalty='l1', loss='l1', dual=False, class_weight='auto'))
            if 'RandomForestClassifier' in tuning_ranges:
                models.append(RandomForestClassifier(oob_score=True))
            if 'GradientBoostingClassifier' in tuning_ranges:
                models.append(GradientBoostingClassifier(subsample=0.5, n_estimators=1000, learning_rate=0.01))

        super(ClassificationSuite, self).__init__(tuning_ranges, models)

        self.scorer = make_scorer(accuracy_score)

class RegressionSuite(BasePredictorSuite):

    def __init__(self, tuning_ranges=None, models=None, standardize=True):
        print 'Not implemented yet!'
        super(RegressionSuite, self).__init__(tuning_ranges, models, standardize)