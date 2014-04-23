import os
import time
import numpy as np
import pandas as pd
import bdtsmc
from truncated_tree_ensemble import get_truncated_shopping_indices
from create_features import make_response_map
from sklearn.cross_validation import KFold
import multiprocessing
from utils import logsumexp, softmax, check_if_zero
from tree_utils import compute_test_metrics_classification, evaluate_predictions_fast

nfolds = 3
ntrunc = 5

def runSmc(args):
    smcData, settings = args
    time_0 = time.clock()
    print '\nInitializing SMC\n'
    # precomputation
    (particles, param, log_weights, cache, cache_tmp) = bdtsmc.init_smc(smcData, settings)
    time_init = time.clock() - time_0

    # Run smc
    print '\nRunning SMC'
    (particles, ess_itr, log_weights_itr, log_pd, particle_stats_itr_d, particles_itr_d, log_pd_islands) = \
            bdtsmc.run_smc(particles, smcData, settings, param, log_weights, cache)
    time_method = time.clock() - time_0     # includes precomputation time
    time_method_sans_init = time.clock() - time_0 - time_init
    
    # Printing some diagnostics
    print
    print 'Estimate of log marginal probability i.e. log p(Y|X) = %s ' % log_pd
    print 'Estimate of log marginal probability for different islands = %s' % log_pd_islands
    print 'logsumexp(log_pd_islands) - np.max(log_pd_islands) = %s\n' % \
            (logsumexp(log_pd_islands) - np.max(log_pd_islands))
    if settings.debug == 1:
        print 'log_weights_itr = \n%s' % log_weights_itr
        # check if log_weights are computed correctly
        for i_, p in enumerate(particles):
            log_w = log_weights_itr[-1, i_] + np.log(settings.n_particles) - np.log(settings.n_islands)
            logprior_p = p.compute_logprior()
            loglik_p = p.compute_loglik()
            logprob_p = p.compute_logprob()
            if (np.abs(settings.ess_threshold) < 1e-15) and (settings.proposal == 'prior'):
                # for the criterion above, only loglik should contribute to the weight update
                try:
                    check_if_zero(log_w - loglik_p)
                except AssertionError:
                    print 'Incorrect weight computation: log_w (smc) = %s, loglik_p = %s' % (log_w, loglik_p)
                    raise AssertionError
            try:
                check_if_zero(logprob_p - loglik_p - logprior_p)
            except AssertionError:
                print 'Incorrect weight computation'
                print 'check if 0: %s, logprior_p = %s, loglik_p = %s' % (logprob_p - loglik_p - logprior_p, logprior_p, loglik_p)
                raise AssertionError

    # Evaluate
    time_predictions_init = time.clock()
    print 'Results on training data (log predictive prob is bogus)'
    # log_predictive on training data is bogus ... you are computing something like \int_{\theta} p(data|\theta) p(\theta|data)
    if settings.weight_islands == 1:
        # each island's prediction is weighted by its marginal likelihood estimate which is equivalent to micro-averaging globally
        weights_prediction = softmax(log_weights_itr[-1, :])
        assert('islandv1' in settings.tag)
    else:
        # correction for macro-averaging predictions across islands
        weights_prediction = np.ones(settings.n_particles) / settings.n_islands
        n_particles_tmp = settings.n_particles / settings.n_islands
        for i_ in range(settings.n_islands):
            pid_min, pid_max = i_ * n_particles_tmp, (i_ + 1) * n_particles_tmp - 1
            pid_range_tmp = range(pid_min, pid_max+1)
            weights_prediction[pid_range_tmp] *= softmax(log_weights_itr[-1, pid_range_tmp]) 
    (pred_prob_overall_train, metrics_train) = \
            evaluate_predictions_smc(particles, smcData, smcData['x_train'], smcData['y_train'], settings, param, weights_prediction)
    print '\nResults on test data'
    (pred_prob_overall_test, metrics_test) = \
            evaluate_predictions_smc(particles, smcData, smcData['x_test'], smcData['y_test'], settings, param, weights_prediction)
    log_prob_train = metrics_train['log_prob']
    log_prob_test = metrics_test['log_prob']
    if settings.optype == 'class':
        acc_train = metrics_train['acc']
        acc_test = metrics_test['acc']
    else:
        mse_train = metrics_train['mse']
        mse_test = metrics_test['mse']
    time_prediction = (time.clock() - time_predictions_init)

    return pred_prob_overall_test, particles, param, weights_prediction

def evaluate_predictions_smc(particles, data, x, y, settings, param, weights):
    if settings.optype == 'class':
        pred_prob_overall = np.zeros((x.shape[0], data['n_class']))
    else:
        pred_prob_overall = np.zeros(x.shape[0])
        pred_mean_overall = np.zeros(x.shape[0])
    if settings.weight_predictions:
        weights_norm = weights
    else:
        weights_norm = np.ones(settings.n_particles) / settings.n_particles
    assert(np.abs(np.sum(weights_norm) - 1) < 1e-3)
    if settings.verbose >= 2:
        print 'weights_norm = '
        print weights_norm
    for pid, p in enumerate(particles):
        pred_all = evaluate_predictions_fast(p, x, y, data, param, settings)
        pred_prob = pred_all['pred_prob']
        pred_prob_overall += weights_norm[pid] * pred_prob
        if settings.optype == 'real':
            pred_mean_overall += weights_norm[pid] * pred_all['pred_mean']
    if settings.debug == 1:
        check_if_zero(np.mean(np.sum(pred_prob_overall, axis=1) - 1))
    if settings.optype == 'class':
        metrics = compute_test_metrics_classification(y, pred_prob_overall)
    else:
        metrics = compute_test_metrics_regression(y, pred_mean_overall, pred_prob_overall)
    if settings.verbose >= 1:
        if settings.optype == 'class':
            print 'Averaging over all particles, accuracy = %f, log predictive = %f' % (metrics['acc'], metrics['log_prob'])
        else:
            print 'Averaging over all particles, mse = %f, log predictive = %f' % (metrics['mse'], metrics['log_prob'])
    return (pred_prob_overall, metrics)

    
def fit_truncated_tree(training_set, response, test_set, n_shop, smcSettings):

    test_cust_ids = test_set.index.get_level_values(0).unique()
    train_cust_ids = training_set.index.get_level_values(0).unique()

    y_predict = pd.DataFrame(data=np.zeros((len(test_cust_ids), 3), dtype=np.int), index=test_cust_ids,
                             columns=['MajorityVote', 'AverageProb', 'LastObsValue'])

    # get CV split
    folds = KFold(len(train_cust_ids), n_folds=nfolds)
    cv_args = []
    cv_scores = []
    training_set = training_set.reset_index(level=1)

    print 'Using features', training_set.columns

    allProbs = {}
    allData = []
    # Separate on customers
    for nFold, (train, validate) in enumerate(folds):
        allProbs[nFold] = []
        
        # reset the index here to include the shopping point as a predictor
        X_train = training_set.ix[train_cust_ids[train]].values
        y_train = response.ix[train_cust_ids[train]].values

        X_val = training_set.ix[train_cust_ids[validate]].values
        y_val = response.ix[train_cust_ids[validate]].values

        # ACB, hopefully not necessary
        X_train[np.where(X_train == False)] = 0.01
        X_train[np.where(X_train == True)] = 0.99
        X_train[np.where(0*X_train != 0)] = 0.0
        X_val[np.where(X_val == False)] = 0.01
        X_val[np.where(X_val == True)] = 0.99
        X_val[np.where(0*X_val != 0)] = 0.0

        for i in range(ntrunc):
            n_shop_train = n_shop[train]
            trunc_idx_train = get_truncated_shopping_indices(n_shop_train)

            n_shop_val = n_shop[validate]
            trunc_idx_val = get_truncated_shopping_indices(n_shop_val)

            # Do it!
            smcData = {}
            smcData["x_train"]   = X_train[trunc_idx_train].astype(np.float)
            smcData["y_train"]   = y_train
            smcData["n_train"]   = smcData["x_train"].shape[0]
            smcData["n_dim"]     = smcData["x_train"].shape[1]
            smcData["n_class"]   = 2304
            smcData["is_sparse"] = False
            smcData["x_test"]    = X_val[trunc_idx_val].astype(np.float)
            smcData["y_test"]    = y_val
            smcData["n_test"]    = smcData["x_test"].shape[0]

            if domultiprocessing:
                allData.append((smcData, smcSettings))
            else:
                allProbs[nFold].append(runSmc((smcData, smcSettings)))
                
    if domultiprocessing:
        results = pool.map(runSmc, allData)
        idx = 0
        for nFold, (train, validate) in enumerate(folds):
            for i in range(ntrunc):
                allProbs[nFold].append(results[idx])
                idx += 1

    for nFold, (train, validate) in enumerate(folds):
        probs = np.array([x[0] for x in allProbs[nFold]])
        psum  = np.sum(probs, axis=0)
        idx   = np.argsort(psum)[:,-1]
        ntot  = len(idx)
        nmat  = np.sum(idx == y_val)
        print "Fold %d: Average over truncated runs yields success rate %d/%d = %.3f" % (nFold, nmat, ntot, 1.0*nmat/ntot)

    #import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    if os.environ["HOST"] == "magneto.astro.washington.edu":
        print "MULTIPROCESSING"
        domultiprocessing = True
        njobs = nfolds*ntrunc
        pool = multiprocessing.Pool(njobs)
        pool.map(int, range(njobs))
    else:
        domultiprocessing = False

    settings = bdtsmc.process_command_line()
    settings.debug = 0
    settings.optype = "class"
    settings.n_particles = 100
    settings.n_islands = max(1, settings.n_particles // 20)
    settings.proposal = "prior"
    settings.grow = "next" # nodewise
    
    training_set = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../data/training_set.h5"), 'df')
    test_set = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../data/test_set.h5"), 'df')
    customer_ids = training_set.index.get_level_values(0).unique()

    response = training_set[training_set['record_type'] == 1]['planID']
    response.index = response.index.get_level_values(0)
    training_set = training_set[training_set['record_type'] == 0]
    n_shop = np.asarray([training_set.ix[cid].index[-1] for cid in customer_ids])

    last_spt_idx = [(cid, test_set.ix[cid].index[-1]) for cid in test_set.index.get_level_values(0).unique()]
    # the next two lines only work when do_each_category = False
    last_plan = test_set.ix[last_spt_idx]['planID']
    last_plan.index = last_plan.index.get_level_values(0)
    not_predictors = ['record_type', 'day', 'state', 'location', 'group_size', 'C_previous', 'planID',
                      'most_common_plan', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    training_set = training_set.drop(not_predictors, axis=1)
    test_set = test_set.drop(not_predictors, axis=1)

    # transform time to be periodic
    training_set['time'] = np.cos(training_set['time'] * np.pi / 12.0)
    test_set['time'] = np.cos(test_set['time'] * np.pi / 12.0)

    # train one big model by not treating the categories as independent
    tstart = time.clock()
    fit_truncated_tree(training_set, response, test_set, n_shop, settings)
    tend = time.clock()
    print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'
