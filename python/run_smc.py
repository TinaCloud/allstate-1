import os
import time
import numpy as np
import pandas as pd
import bdtsmc
from truncated_tree_ensemble import get_truncated_shopping_indices
from sklearn.cross_validation import KFold
import multiprocessing

njobs = 8
pool = multiprocessing.Pool(njobs)
pool.map(int, range(njobs))
nfolds = 3
ntrunc = 5
domultiprocessing = False

def runSmc(args):
    smcData, smcSettings = args
    time_0 = time.clock()
    print '\nInitializing SMC\n'
    # precomputation
    (particles, param, log_weights, cache, cache_tmp) = bdtsmc.init_smc(smcData, smcSettings)
    time_init = time.clock() - time_0

    # Run smc
    print '\nRunning SMC'
    (particles, ess_itr, log_weights_itr, log_pd, particle_stats_itr_d, particles_itr_d, log_pd_islands) = \
            bdtsmc.run_smc(particles, smcData, smcSettings, param, log_weights, cache)
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
            evaluate_predictions_smc(particles, smcData, smcData['x_train'], smcData['y_train'], smcSettings, param, weights_prediction)
    print '\nResults on test data'
    (pred_prob_overall_test, metrics_test) = \
            evaluate_predictions_smc(particles, smcData, smcData['x_test'], smcData['y_test'], smcSettings, param, weights_prediction)
    log_prob_train = metrics_train['log_prob']
    log_prob_test = metrics_test['log_prob']
    if settings.optype == 'class':
        acc_train = metrics_train['acc']
        acc_test = metrics_test['acc']
    else:
        mse_train = metrics_train['mse']
        mse_test = metrics_test['mse']
    time_prediction = (time.clock() - time_predictions_init)

    return pred_prob_overall_test
    
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
    for train, validate in folds:
        allProbs[folds] = []
        
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
                pred_prob_overall_test = runSmc((smcData, smcSettings))
                allProbs[folds].append(pred_prob_overall_test)
                
    if domultiprocessing:
        results = pool.map(runSmc, allData)
        idx = 0
        for train, validate in folds:
            for i in range(ntrunc):
                allProbs[folds].append(results[idx])
                idx += 1

    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    settings = bdtsmc.process_command_line()
    settings.debug = 1
    settings.optype = "class"
    settings.n_particles = 20
    settings.n_islands = settings.n_particles // 20
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
    prediction, trunc_tree = fit_truncated_tree(training_set, response, test_set, n_shop, settings)
    tend = time.clock()
    print '\n', 'Training took', (tend - tstart) / 3600.0, 'hours.', '\n'
