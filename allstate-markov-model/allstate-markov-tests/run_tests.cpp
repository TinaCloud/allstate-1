//
//  main.cpp
//  allstate-markov-tests
//
//  Created by Brandon Kelly on 5/16/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#define CATCH_CONFIG_MAIN

#include <iostream>
#include <cmath>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <string>

// local includes
#include <catch.hpp>
#include <samplers.hpp>
#include <steps.hpp>
#include "cluster.hpp"
#include "categorical.hpp"
#include "unbounded_counts.hpp"
#include "bounded_counts.hpp"
#include "markov_chain.hpp"

using boost::math::lgamma;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;


bool approximately_equal(double a, double b, double eps=1e-6, double rtol=1e-4) {
    bool pass = false;
    double diff = std::abs(a - b);
    double rdiff = std::abs(a - b) / (std::abs(a) + std::abs(b));
    if (diff < eps) {
        pass = true;
    } else if (rdiff < rtol) {
        pass = true;
    }
    return pass;
}

arma::uvec generate_cluster_labels(int ndata, std::vector<double> pi) {
    arma::uvec zvalues(ndata);
    
    boost::random::discrete_distribution<> categorical_distribution(pi.begin(), pi.end());
    
    for (int i=0; i<ndata; i++) {
        zvalues(i) = categorical_distribution(rng);
    }
    
    return zvalues;
}

arma::uvec generate_bounded_counts(arma::uvec zlabels, arma::vec probs, int nmax) {
    arma::uvec counts(zlabels.n_elem);
    std::vector<boost::random::binomial_distribution<> > distributions;
    for (int j=0; j<probs.n_elem; j++) {
        distributions.push_back(boost::random::binomial_distribution<> (nmax, probs(j)));
    }
    for (int i=0; i<zlabels.n_elem; i++) {
        counts(i) = distributions[zlabels(i)](rng);
    }
    return counts;
}

arma::uvec generate_unbounded_counts(arma::uvec zlabels, arma::vec probs, int rfailures) {
    arma::uvec counts(zlabels.n_elem);
    std::vector<boost::random::negative_binomial_distribution<> > distributions;
    for (int j=0; j<probs.n_elem; j++) {
        // boost syntax wants probability of failure, we input probability of success
        distributions.push_back(boost::random::negative_binomial_distribution<> (rfailures, 1.0 - probs(j)));
    }
    for (int i=0; i<zlabels.n_elem; i++) {
        counts(i) = distributions[zlabels(i)](rng);
    }
    return counts;
}

arma::uvec generate_categoricals(arma::uvec zlabels, arma::mat probs) {
    int ndata = zlabels.n_elem;
    int nclusters = probs.n_rows;
    
    std::vector<boost::random::discrete_distribution<> > cat_dists;
    for (int k=0; k<nclusters; k++) {
        std::vector<double> probs_k = arma::conv_to<std::vector<double> >::from(probs.row(k));
        cat_dists.push_back(boost::random::discrete_distribution<> (probs_k.begin(), probs_k.end()));
    }
    arma::uvec categories(ndata);
    for (int i=0; i<ndata; i++) {
        categories(i) = cat_dists[zlabels(i)](rng);
    }
    return categories;
}

std::vector<std::vector<int> > generate_markov_chain(int nchains, arma::mat Tprob) {
    int nstates = Tprob.n_cols;
    arma::uvec ntime = arma::randi<arma::uvec>(nchains, arma::distr_param(2, 7));
    
    std::vector<std::vector<double> > Tprob_std;
    for (int r=0; r<nstates; r++) {
        Tprob_std.push_back(arma::conv_to<std::vector<double> >::from(Tprob.row(r)));
    }
    
    std::vector<std::vector<int> > chains(nchains);
    boost::random::uniform_int_distribution<> istate(0, nstates-1);
    for (int i=0; i<nchains; i++) {
        std::vector<int> chain_i(ntime(i));
        chain_i[0] = istate(rng);
        for (int t=1; t<ntime(i); t++) {
            int previous_state = chain_i[t-1];
            int next_state =
                boost::random::discrete_distribution<>
                (Tprob_std[previous_state].begin(), Tprob_std[previous_state].end())(rng);
            chain_i[t] = next_state;
        }
        chains[i] = chain_i;
    }
    return chains;
}

TEST_CASE("Set the seed for the random number generator for reproducibility.", "[startup]") {
    rng.seed(123456);
}

// make sure the categorical distribution return the right fractions
TEST_CASE("Test categorical distribution generator.", "[categorical]") {
    int ndata = 100000;
    double pi[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi_v(5);
    for (int i=0; i<5; i++) {
        pi_v[i] = pi[i];
    }
    arma::uvec zvalues = generate_cluster_labels(ndata, pi_v);
    arma::vec zcounts(5);
    for (int i=0; i<ndata; i++) {
        zcounts(zvalues(i))++;
    }
    for (int j=0; j<5; j++) {
        double counts_mean = pi[j] * ndata;
        double counts_var = counts_mean * (1.0 - pi[j]);
        double zscore = (zcounts(j) - counts_mean) / sqrt(counts_var);
        REQUIRE(abs(zscore) < 3.0);
    }
}

TEST_CASE("Test counting methods for cluster labels class.", "[cluster labels]") {
    int ndata = 10000;
    double pi0[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi(5);
    for (int i=0; i<5; i++) {
        pi[i] = pi0[i];
    }
    
    arma::uvec zlabels0 = generate_cluster_labels(ndata, pi);
    
    std::shared_ptr<ClusterLabels> zlabels =
    std::make_shared<ClusterLabels>(ClusterLabels(true, "Z", ndata, 5));
    
    // first test counts the cluster labels
    zlabels->Save(zlabels0);
    zlabels->CountClusters();
    
    arma::vec zcounts = arma::zeros<arma::vec>(pi.size());
    for (int i=0; i<ndata; i++) {
        zcounts(zlabels0(i))++;
    }
    
    for (int k=0; k<pi.size(); k++) {
        double zc1 = zcounts(k);
        double zc2 = zlabels->GetClusterCounts()(k);
        REQUIRE(approximately_equal(zc1, zc2));
    }

    // generate some categorical data
    arma::mat probs = arma::randu<arma::mat>(5, 4);  // 4 categories
    for (int k=0; k<5; k++) {
        probs.row(k) /= arma::sum(probs.row(k));
    }
    
    arma::uvec categories = generate_categoricals(zlabels0, probs);
    
    zlabels->AddCategoricalPop(std::make_shared<CategoricalPop>(false, "categories", categories));
    
    // now count the categories for each cluster
    zlabels->CountCategories();
    
    arma::mat category_counts = arma::zeros<arma::mat>(pi.size(), probs.n_cols);
    for (int i=0; i<ndata; i++) {
        category_counts(zlabels0(i), categories(i))++;
    }
    
    for (int k=0; k<pi.size(); k++) {
        for (int j=0; j<probs.n_cols; j++) {
            double n_kj1 = category_counts(k,j);
            double n_kj2 = zlabels->GetCategoryCounts(0)(k,j);
            REQUIRE(approximately_equal(n_kj1, n_kj2));
        }
    }
    
    arma::mat n_kj = zlabels->GetCategoryCounts(0);
    for (int k=0; k<pi.size(); k++) {
        double colsum = arma::sum(n_kj.row(k));
        REQUIRE(approximately_equal(colsum, zcounts(k)));
    }
    
}

// test the methods of the bounded counts object
TEST_CASE("Test methods of bounded count object.", "[bounded counts]") {
    // generate some data
    int ndata = 10000;
    double pi0[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi(5);
    for (int i=0; i<5; i++) {
        pi[i] = pi0[i];
    }

    arma::uvec zlabels0 = generate_cluster_labels(ndata, pi);
    
    std::shared_ptr<ClusterLabels> zlabels =
        std::make_shared<ClusterLabels>(ClusterLabels(true, "Z", ndata, 5));
    
    zlabels->Save(zlabels0);
    zlabels->CountClusters();
    
    int nmax = 100;
    arma::vec probs = {0.1, 0.2, 0.3, 0.05, 0.95};
    arma::uvec counts = generate_bounded_counts(zlabels->Value(), probs, nmax);
    
    BoundedCountsPop AlphaBeta(true, "bcounts", counts, nmax);
    
    AlphaBeta.SetClusterLabels(zlabels);
    // make sure point is correctly set
    REQUIRE(AlphaBeta.GetClusterLabels() == zlabels);
    
    // test the LogDensity method by making sure ratios are correct when computed two different ways
    arma::vec alpha1 = {0.1, 7.0};
    arma::vec log_alpha1 = arma::log(alpha1);
    arma::vec alpha2 = {14.0, 0.01};
    arma::vec log_alpha2 = arma::log(alpha2);
    
    double logratio = AlphaBeta.LogDensity(log_alpha1) - AlphaBeta.LogDensity(log_alpha2);
    
    // now compute the ratio the slow way, including terms that cancel
    double logratio_slow1 = (AlphaBeta.prior_shape - 1.0) * arma::sum(log_alpha1) - arma::sum(alpha1) / AlphaBeta.prior_scale;
    double logratio_slow2 = (AlphaBeta.prior_shape - 1.0) * arma::sum(log_alpha2) - arma::sum(alpha2) / AlphaBeta.prior_scale;
    arma::vec counts_sum = arma::zeros<arma::vec>(5);
    arma::vec zcounts = arma::zeros<arma::vec>(5);
    for (int i=0; i<ndata; i++) {
        logratio_slow1 += -log(nmax + 1.0) - lbeta(nmax - counts(i) + 1.0, counts(i) + 1.0);
        logratio_slow2 += -log(nmax + 1.0) - lbeta(nmax - counts(i) + 1.0, counts(i) + 1.0);
        counts_sum(zlabels0(i)) += counts(i);
        zcounts(zlabels0(i))++;
    }
    for (int k=0; k<zcounts.n_elem; k++) {
        logratio_slow1 += lbeta(alpha1(0) + counts_sum(k),
                                zcounts(k) * nmax + alpha1(1) - counts_sum(k));
        logratio_slow1 -= lbeta(alpha1(0), alpha1(1));
        logratio_slow2 += lbeta(alpha2(0) + counts_sum(k),
                                zcounts(k) * nmax + alpha2(1) - counts_sum(k));
        logratio_slow2 -= lbeta(alpha2(0), alpha2(1));
    }
    double logratio_slow = logratio_slow1 - logratio_slow2;
    
    REQUIRE(approximately_equal(logratio, logratio_slow));
}

TEST_CASE("Test methods of unbounded counts object.", "[unbounded counts]") {
    // generate some data
    int ndata = 10000;
    double pi0[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi(5);
    for (int i=0; i<5; i++) {
        pi[i] = pi0[i];
    }
    
    arma::uvec zlabels0 = generate_cluster_labels(ndata, pi);
    
    std::shared_ptr<ClusterLabels> zlabels = std::make_shared<ClusterLabels>(ClusterLabels(true, "Z", ndata, 5));
    
    zlabels->Save(zlabels0);
    zlabels->CountClusters();
    
    arma::vec probs = {0.1, 0.2, 0.3, 0.05, 0.95};
    int rfailures = 20;
    arma::uvec counts = generate_unbounded_counts(zlabels->Value(), probs, rfailures);
    
    UnboundedCountsPop AlphaBetaR(true, "ubcounts", counts);
    
    AlphaBetaR.SetClusterLabels(zlabels);
    // make sure point is correctly set
    REQUIRE(AlphaBetaR.GetClusterLabels() == zlabels);
    
    // test the LogDensity method by making sure ratios are correct when computed two different ways
    arma::vec alpha1 = {0.1, 7.0, 5.0};
    arma::vec log_alpha1 = arma::log(alpha1);
    arma::vec alpha2 = {14.0, 0.01, 65.0};
    arma::vec log_alpha2 = arma::log(alpha2);
    
    double logratio = AlphaBetaR.LogDensity(log_alpha1) - AlphaBetaR.LogDensity(log_alpha2);
    
    // now compute the ratio the slow way, including terms that cancel
    double logratio_slow1 = (AlphaBetaR.prior_ashape - 1.0) * (log_alpha1(0) + log_alpha1(1)) - (alpha1(0) + alpha1(1)) / AlphaBetaR.prior_ascale + (AlphaBetaR.prior_rshape - 1.0) * log_alpha1(2) - alpha1(2) / AlphaBetaR.prior_rscale;
    double logratio_slow2 = (AlphaBetaR.prior_ashape - 1.0) * (log_alpha2(0) + log_alpha2(1)) - (alpha2(0) + alpha2(1)) / AlphaBetaR.prior_ascale + (AlphaBetaR.prior_rshape - 1.0) * log_alpha2(2) - alpha2(2) / AlphaBetaR.prior_rscale;
    
    // add in the normalization
    for (int i=0; i<ndata; i++) {
        logratio_slow1 += lgamma(counts(i) + alpha1(2)) - lgamma(counts(i) + 1.0) - lgamma(alpha1(2));
        logratio_slow2 += lgamma(counts(i) + alpha2(2)) - lgamma(counts(i) + 1.0) - lgamma(alpha2(2));
    }

    arma::vec counts_sum = arma::zeros<arma::vec>(5);
    arma::vec zcounts = arma::zeros<arma::vec>(5);
    for (int i=0; i<ndata; i++) {
        counts_sum(zlabels0(i)) += counts(i);
        zcounts(zlabels0(i))++;
    }
    
    for (int k=0; k<zcounts.n_elem; k++) {
        logratio_slow1 += lbeta(alpha1(0) + counts_sum(k), alpha1(1) + zcounts(k) * alpha1(2)) - lbeta(alpha1(0), alpha1(1));
        logratio_slow2 += lbeta(alpha2(0) + counts_sum(k), alpha2(1) + zcounts(k) * alpha2(2)) - lbeta(alpha2(0), alpha2(1));
    }
    double logratio_slow = logratio_slow1 - logratio_slow2;
    
    REQUIRE(approximately_equal(logratio, logratio_slow));
}

TEST_CASE("Test methods of Categorical variables.", "[categorical]") {
    int ndata = 10000;
    double pi0[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi(5);
    for (int i=0; i<5; i++) {
        pi[i] = pi0[i];
    }
    
    arma::uvec zlabels0 = generate_cluster_labels(ndata, pi);
    
    std::shared_ptr<ClusterLabels> zlabels = std::make_shared<ClusterLabels>(ClusterLabels(true, "Z", ndata, 5));
    
    // first test counts the cluster labels
    zlabels->Save(zlabels0);
    zlabels->CountClusters();
    
    // generate some categorical data
    arma::mat probs = arma::randu<arma::mat>(5, 4);  // 4 categories
    for (int k=0; k<5; k++) {
        probs.row(k) /= arma::sum(probs.row(k));
    }
    
    arma::uvec categories = generate_categoricals(zlabels0, probs);
    
    std::shared_ptr<CategoricalPop> categorical = std::make_shared<CategoricalPop>(false, "categorical", categories);
    
    REQUIRE(categorical->ndata == ndata);
    REQUIRE(categorical->GetNcategories() == probs.n_cols);
    
    zlabels->AddCategoricalPop(categorical);
    categorical->SetClusterLabels(zlabels);
    
    // make sure pointer is correctly set
    REQUIRE(categorical->GetClusterLabels() == zlabels);
    REQUIRE(categorical->GetIndex() == 0);
    
    // now count the categories for each cluster
    zlabels->CountCategories();
    
    // make sure ratio of values posteriors agrees with slow calculation
    arma::vec log_alpha1 = categorical->StartingValue();
    arma::vec log_alpha2 = categorical->StartingValue();
    double logratio = categorical->LogDensity(log_alpha1) - categorical->LogDensity(log_alpha2);
    arma::vec alpha1 = arma::exp(log_alpha1);
    arma::vec alpha2 = arma::exp(log_alpha2);
    
    // start with prior
    double logratio_slow1 = (categorical->prior_shape - 1.0) * arma::sum(log_alpha1) - arma::sum(alpha1) / categorical->prior_scale;
    double logratio_slow2 = (categorical->prior_shape - 1.0) * arma::sum(log_alpha2) - arma::sum(alpha2) / categorical->prior_scale;
    
    arma::mat n_kj = zlabels->GetCategoryCounts(categorical->GetIndex());
    for (int k=0; k<pi.size(); k++) {
        logratio_slow1 += lgamma(arma::sum(alpha1));
        logratio_slow2 += lgamma(arma::sum(alpha2));
        double n_a_sum1 = 0.0;
        double n_a_sum2 = 0.0;
        for (int j=0; j<categorical->GetNcategories(); j++) {
            logratio_slow1 += lgamma(n_kj(k,j) + alpha1(j)) - lgamma(alpha1(j));
            logratio_slow2 += lgamma(n_kj(k,j) + alpha2(j)) - lgamma(alpha2(j));
            n_a_sum1 += alpha1(j) + n_kj(k,j);
            n_a_sum2 += alpha2(j) + n_kj(k,j);
        }
        logratio_slow1 -= lgamma(n_a_sum1);
        logratio_slow2 -= lgamma(n_a_sum2);
    }

    double logratio_slow = logratio_slow1 - logratio_slow2;
    REQUIRE(approximately_equal(logratio, logratio_slow));
}

TEST_CASE("Test methods of TransitionProb object.", "[transition prob]") {
    // generate some data
    int ndata = 10000;
    int nstates = 3;
    int nclusters = 2;
    
    std::vector<double> pi(2);
    pi[0] = 0.35;
    pi[1] = 0.65;
    
    arma::uvec zlabels0 = generate_cluster_labels(ndata, pi);

    arma::mat Tprob0 = arma::randu<arma::mat>(nstates, nstates);
    // row sums must be normalized to one
    for (int r=0; r<nstates; r++) {
        Tprob0.row(r) /= arma::sum(Tprob0.row(r));
    }
    
    std::vector<std::vector<int> > chains = generate_markov_chain(ndata, Tprob0);
    
    int cluster_id = 1;
    TransitionProbability tprob_object(true, "Tprob", chains, nstates, cluster_id);
    
    // first make sure cluster labels pointer is correctly set
    std::shared_ptr<ClusterLabels> p_labels = std::make_shared<ClusterLabels>(true, "Z", ndata, nclusters);
    p_labels->Save(zlabels0);
    
    tprob_object.SetClusterLabels(p_labels);
    
    REQUIRE(tprob_object.GetClusterLabels() == p_labels);
    
    // now add the population-level parameters and make sure their pointer are correctly set
    for (int r=0; r<nstates; r++) {
        for (int c=0; c<nstates; c++) {
            std::shared_ptr<TransitionPopulation> p_tpop = std::make_shared<TransitionPopulation>(true, "Tpop", r, c);
            tprob_object.AddPopulationPtr(p_tpop);
            REQUIRE(p_tpop == tprob_object.GetPopulationPtr(r, c));
        }
    }
    
    // now generate draws from the posterior and make sure these agree with the input transition matrix
    int ndraws = 1000;
    arma::mat Tmean = arma::zeros<arma::mat>(nstates, nstates);
    for (int i=0; i<ndraws; i++) {
        arma::mat Tdraw = tprob_object.RandomPosterior();
        Tmean = Tmean + Tdraw;
    }
    Tmean /= ndraws;
    
    for (int r=0; r<nstates; r++) {
        for (int c=0; c<nstates; c++) {
            CHECK(approximately_equal(Tprob0(r,c), Tmean(r,c), 0.01, 0.05));
        }
    }
}

TEST_CASE("Test methods of TransitionPopulation class", "[transition population]") {
    // first generate some transition matrices
    int nstates = 5;
    int nclusters = 10;
    
    std::vector<arma::mat> Tprobs(nclusters);
    for (int k=0; k<nclusters; k++) {
        arma::mat Tprob_k = arma::randu<arma::mat>(nstates, nstates);
        // row sums must be normalized to one
        for (int r=0; r<nstates; r++) {
            Tprob_k.row(r) /= arma::sum(Tprob_k.row(r));
        }
        Tprobs[k] = Tprob_k;
    }

    std::vector<std::shared_ptr<TransitionPopulation> > gammas;
    for (int r=0; r<nstates; r++) {
        for (int c=0; c<nstates; c++) {
            gammas.push_back(std::make_shared<TransitionPopulation>(true, "gamma", r, c));
        }
    }

    int ndata = 100;
    std::vector<std::vector<int> > chains = generate_markov_chain(ndata, Tprobs[0]);
    
    std::vector<std::shared_ptr<TransitionProbability> > tprob_objects(nclusters);
    for (int k=0; k<nclusters; k++) {
        tprob_objects[k] = std::make_shared<TransitionProbability>(true, "T", chains, nstates, k);
        tprob_objects[k]->Save(Tprobs[k]);
    }
    std::shared_ptr<TransitionHyperPrior> hyper = std::make_shared<TransitionHyperPrior>(true, "hyper");
    
    // make sure the population-level object know about the cluster-level and hyper prior
    for (int i=0; i<gammas.size(); i++) {
        for (int k=0; k<nclusters; k++) {
            gammas[i]->AddTransitionMatrix(tprob_objects[k]);
        }
        gammas[i]->SetHyperPrior(hyper);
        // add the other gammas in this row for this object
        for (int j=0; j<gammas.size(); j++) {
            if (j != i && gammas[j]->row_idx == gammas[i]->row_idx) {
                gammas[i]->AddGamma(gammas[j]);
            }
        }
    }
    
    // now make sure all the pointers are correctly set
    for (int i=0; i<gammas.size(); i++) {
        for (int k=0; k<nclusters; k++) {
            REQUIRE(tprob_objects[k] == gammas[i]->GetTransitionMatrix(k));
        }
        REQUIRE(hyper == gammas[i]->GetHyperPrior());
        for (int j=0; j<gammas.size(); j++) {
            if (j != i && gammas[j]->row_idx == gammas[i]->row_idx) {
                REQUIRE(gammas[i]->GetGamma(gammas[j]->col_idx) == gammas[j]);
            }
        }
    }
    
    // finally, compare log-densities. just do this for a single row
    int test_row = 2;
    int test_col = 3;
    int g_idx = 0;
    hyper->Save(hyper->StartingValue());
    
    for (int i=0; i<gammas.size(); i++) {
        if (gammas[i]->row_idx == test_row && gammas[i]->col_idx == test_col) {
            g_idx = i;
        }
        gammas[i]->Save(gammas[i]->StartingValue());
    }
    
    double test_gamma = 3.4;
    double old_gamma = exp(gammas[g_idx]->Value());
    double logratio = gammas[g_idx]->LogDensity(log(test_gamma)) - gammas[g_idx]->LogDensity((log(old_gamma)));
    
    // calculate the ratio of log densities the slow the way
    double logratio_slow1 = -0.5 * (log(test_gamma) - hyper->Value()(0)) * (log(test_gamma) - hyper->Value()(0)) / hyper->Value()(1);
    double logratio_slow2 = -0.5 * (log(old_gamma) - hyper->Value()(0)) * (log(old_gamma) - hyper->Value()(0)) / hyper->Value()(1);

    for (int k=0; k<nclusters; k++) {
        gammas[g_idx]->Save(log(test_gamma));
        double gamma_sum = 0.0;
        for (int i=0; i<gammas.size(); i++) {
            if (gammas[i]->row_idx == test_row) {
                // sum over columns for this row
                gamma_sum += exp(gammas[i]->Value());
                logratio_slow1 -= lgamma(exp(gammas[i]->Value()));
                logratio_slow1 += (exp(gammas[i]->Value()) - 1.0) * log(Tprobs[k](gammas[i]->row_idx, gammas[i]->col_idx));
            }
        }
        logratio_slow1 += lgamma(gamma_sum);
        
        gammas[g_idx]->Save(log(old_gamma));
        gamma_sum = 0.0;
        for (int i=0; i<gammas.size(); i++) {
            if (gammas[i]->row_idx == test_row) {
                // sum over columns for this row
                gamma_sum += exp(gammas[i]->Value());
                logratio_slow2 -= lgamma(exp(gammas[i]->Value()));
                logratio_slow2 += (exp(gammas[i]->Value()) - 1.0) * log(Tprobs[k](gammas[i]->row_idx, gammas[i]->col_idx));
            }
        }
        logratio_slow2 += lgamma(gamma_sum);
    }
    
    double logratio_slow = logratio_slow1 - logratio_slow2;
    REQUIRE(approximately_equal(logratio_slow, logratio));
}

TEST_CASE("Test methods of ClusterLabels class, sans the LogDensity methods.", "[cluster labels]") {
    // instantiate the cluster labels class
    int ndata = 10000;
    int nclusters = 5;
    
    ClusterLabels cluster(false, "Z", ndata, nclusters);
    
    // first test the StartingValue method
    arma::uvec zlabels = cluster.StartingValue();
    arma::vec zcounts = cluster.GetClusterCounts();
    for (int k=0; k<nclusters; k++) {
        REQUIRE(zcounts(k) > 0);
    }
    REQUIRE(zlabels.min() == 0);
    REQUIRE(zlabels.max() == (nclusters - 1));
    
    // generate some data and instantiate the corresponding objects
    
    // categorical data
    arma::mat probs1 = arma::randu<arma::mat>(nclusters, 4);  // 4 categories
    for (int k=0; k<nclusters; k++) {
        probs1.row(k) /= arma::sum(probs1.row(k));
    }
    
    arma::uvec categories1 = generate_categoricals(zlabels, probs1);
    
    arma::mat probs2 = arma::randu<arma::mat>(nclusters, 6);  // 6 categories
    for (int k=0; k<nclusters; k++) {
        probs2.row(k) /= arma::sum(probs2.row(k));
    }
    
    arma::uvec categories2 = generate_categoricals(zlabels, probs2);

    std::shared_ptr<CategoricalPop> catvar1 = std::make_shared<CategoricalPop>(true, "CAT-1", categories1);
    std::shared_ptr<CategoricalPop> catvar2 = std::make_shared<CategoricalPop>(true, "CAT-2", categories2);
    
    cluster.AddCategoricalPop(catvar1);
    cluster.AddCategoricalPop(catvar2);
    
    // make sure pointers are correctly set
    std::vector<std::shared_ptr<CategoricalPop> > p_categoricals = cluster.GetCategoricals();
    REQUIRE(p_categoricals.size() == 2);
    for (int l=0; l<p_categoricals.size(); l++) {
        REQUIRE(p_categoricals[l]->GetIndex() == l);
    }
    REQUIRE(p_categoricals[0] == catvar1);
    REQUIRE(p_categoricals[1] == catvar2);
    
    cluster.CountCategories();
    
    // make sure counts are updated when we remove a cluster label
    int data_idx = 13;
    arma::mat catcounts = cluster.GetCategoryCounts(1);
    REQUIRE(catcounts.n_rows == nclusters);
    REQUIRE(catcounts.n_cols == probs2.n_cols);
    std::vector<int> category_values = cluster.RemoveClusterLabel(data_idx);
    REQUIRE(category_values[0] == categories1(data_idx));
    REQUIRE(category_values[1] == categories2(data_idx));
    
    arma::vec new_zcounts = cluster.GetClusterCounts();
    for (int k=0; k<nclusters; k++) {
        if (zlabels(data_idx) == k) {
            REQUIRE(new_zcounts(k) == zcounts(k) - 1);
        } else {
            REQUIRE(new_zcounts(k) == zcounts(k));
        }
    }
    
    arma::mat catcounts_diff = catcounts - cluster.GetCategoryCounts(1);
    int nzeros = 0;
    for (int k=0; k<nclusters; k++) {
        for (int j=0; j<catcounts_diff.n_cols; j++) {
            if (zlabels(data_idx) == k && categories2(data_idx) == j) {
                REQUIRE(approximately_equal(catcounts_diff(k,j), 1.0));
            } else nzeros += approximately_equal(catcounts_diff(k,j), 0.0);
        }
    }
    REQUIRE(nzeros == catcounts_diff.n_elem - 1);
    
    // make sure counts are updated when we add the cluster label back in
    cluster.UpdateClusterCounts(data_idx, zlabels(data_idx));
    arma::vec zcounts_diff = cluster.GetClusterCounts() - zcounts;
    REQUIRE(approximately_equal(0.0, arma::sum(zcounts_diff)));
    
    catcounts_diff = catcounts - cluster.GetCategoryCounts(1);
    REQUIRE(approximately_equal(0.0, arma::accu(catcounts_diff)));
    
    // bounded counts data
    int nmax1 = 100;
    int nmax2 = 23;
    arma::vec p1 = arma::randu<arma::vec>(nclusters);
    arma::vec p2 = arma::randu<arma::vec>(nclusters);
    arma::uvec bcounts1 = generate_bounded_counts(zlabels, p1, nmax1);
    arma::uvec bcounts2 = generate_bounded_counts(zlabels, p2, nmax2);
    
    std::shared_ptr<BoundedCountsPop> bcvar1 = std::make_shared<BoundedCountsPop>(true, "B-1", bcounts1, nmax1);
    std::shared_ptr<BoundedCountsPop> bcvar2 = std::make_shared<BoundedCountsPop>(true, "B-2", bcounts2, nmax2);
    
    cluster.AddBoundedCountsPop(bcvar1);
    cluster.AddBoundedCountsPop(bcvar2);
    std::vector<std::shared_ptr<BoundedCountsPop> > p_bounded_counts = cluster.GetBoundedCounts();
    REQUIRE(p_bounded_counts.size() == 2);
    REQUIRE(p_bounded_counts[0] == bcvar1);
    REQUIRE(p_bounded_counts[1] == bcvar2);
    
    // unbounded counts data
    int rfail1 = 100;
    int rfail2 = 23;
    int rfail3 = 234;
    p1 = arma::randu<arma::vec>(nclusters);
    p2 = arma::randu<arma::vec>(nclusters);
    arma::vec p3 = arma::randu<arma::vec>(nclusters);
    arma::uvec ubcounts1 = generate_bounded_counts(zlabels, p1, rfail1);
    arma::uvec ubcounts2 = generate_bounded_counts(zlabels, p2, rfail2);
    arma::uvec ubcounts3 = generate_bounded_counts(zlabels, p3, rfail3);
    
    std::shared_ptr<UnboundedCountsPop> ubcvar1 = std::make_shared<UnboundedCountsPop>(true, "UB-1", ubcounts1);
    std::shared_ptr<UnboundedCountsPop> ubcvar2 = std::make_shared<UnboundedCountsPop>(true, "UB-2", ubcounts2);
    std::shared_ptr<UnboundedCountsPop> ubcvar3 = std::make_shared<UnboundedCountsPop>(true, "UB-3", ubcounts3);
    
    cluster.AddUnboundedCountsPop(ubcvar1);
    cluster.AddUnboundedCountsPop(ubcvar2);
    cluster.AddUnboundedCountsPop(ubcvar3);
    std::vector<std::shared_ptr<UnboundedCountsPop> > p_unbounded_counts = cluster.GetUnboundedCounts();
    REQUIRE(p_unbounded_counts.size() == 3);
    REQUIRE(p_unbounded_counts[0] == ubcvar1);
    REQUIRE(p_unbounded_counts[1] == ubcvar2);
    REQUIRE(p_unbounded_counts[2] == ubcvar3);
    
    // markov transition matrices
    int nstates = 4;
    arma::mat Tprob0 = arma::randu<arma::mat>(nstates, nstates);
    // row sums must be normalized to one
    for (int r=0; r<nstates; r++) {
        Tprob0.row(r) /= arma::sum(Tprob0.row(r));
    }
    
    std::vector<std::vector<int> > chains = generate_markov_chain(ndata, Tprob0);
    
    std::vector<std::shared_ptr<TransitionProbability> > p_Tmats;
    for (int k=0; k<nclusters; k++) {
        std::shared_ptr<TransitionProbability> p_Tmat = std::make_shared<TransitionProbability>(true, "T", chains, nstates, k);
        p_Tmats.push_back(p_Tmat);
        cluster.AddTransitionMatrix(p_Tmat);
    }
    for (int k=0; k<nclusters; k++) {
        REQUIRE(cluster.GetTransitionMatrices()[k] == p_Tmats[k]);
    }
    
}

// test the methods of ClusterLabels assocated with computing the conditional probabilities
TEST_CASE("Test conditional probabilities of ClusterLabels class.", "[cluster labels]") {
    // first generate the data
    int ndata = 10000;
    double pi0[5] = {0.1, 0.3, 0.05, 0.45, 0.1};
    std::vector<double> pi(5);
    for (int i=0; i<5; i++) {
        pi[i] = pi0[i];
    }
    int nclusters = pi.size();
    
    ClusterLabels cluster(false, "Z", ndata, nclusters);
    cluster.Save(generate_cluster_labels(ndata, pi));
    
    // categorical data
    arma::mat probs1 = arma::randu<arma::mat>(nclusters, 4);  // 4 categories
    for (int k=0; k<nclusters; k++) {
        probs1.row(k) /= arma::sum(probs1.row(k));
    }
    
    arma::uvec categories1 = generate_categoricals(cluster.Value(), probs1);
    
    arma::mat probs2 = arma::randu<arma::mat>(nclusters, 6);  // 6 categories
    for (int k=0; k<nclusters; k++) {
        probs2.row(k) /= arma::sum(probs2.row(k));
    }
    
    arma::uvec categories2 = generate_categoricals(cluster.Value(), probs2);
    
    // bounded counts data
    int nmax1 = 100;
    int nmax2 = 23;
    arma::vec p1 = arma::randu<arma::vec>(nclusters);
    arma::vec p2 = arma::randu<arma::vec>(nclusters);
    arma::uvec bcounts1 = generate_bounded_counts(cluster.Value(), p1, nmax1);
    arma::uvec bcounts2 = generate_bounded_counts(cluster.Value(), p2, nmax2);
    
    // unbounded counts data
    int rfail1 = 100;
    int rfail2 = 23;
    int rfail3 = 234;
    p1 = arma::randu<arma::vec>(nclusters);
    p2 = arma::randu<arma::vec>(nclusters);
    arma::vec p3 = arma::randu<arma::vec>(nclusters);
    arma::uvec ubcounts1 = generate_bounded_counts(cluster.Value(), p1, rfail1);
    arma::uvec ubcounts2 = generate_bounded_counts(cluster.Value(), p2, rfail2);
    arma::uvec ubcounts3 = generate_bounded_counts(cluster.Value(), p3, rfail3);
    
    // set the parameter objects
    cluster.AddCategoricalPop(std::make_shared<CategoricalPop>(true, "CAT-1", categories1));
    cluster.AddCategoricalPop(std::make_shared<CategoricalPop>(true, "CAT-2", categories2));
    cluster.AddBoundedCountsPop(std::make_shared<BoundedCountsPop>(true, "B-1", bcounts1, nmax1));
    cluster.AddBoundedCountsPop(std::make_shared<BoundedCountsPop>(true, "B-2", bcounts2, nmax2));
    cluster.AddUnboundedCountsPop(std::make_shared<UnboundedCountsPop>(true, "UB-1", ubcounts1));
    cluster.AddUnboundedCountsPop(std::make_shared<UnboundedCountsPop>(true, "UB-2", ubcounts2));
    cluster.AddUnboundedCountsPop(std::make_shared<UnboundedCountsPop>(true, "UB-3", ubcounts3));

    cluster.CountClusters();
    cluster.CountCategories();
    
    int test_idx = ndata / 4;
    // need to remove this test point
    std::vector<int> test_categories = cluster.RemoveClusterLabel(test_idx);
    
    arma::vec zcounts = cluster.GetClusterCounts();
    
    /*
     *  first make sure marginal distributions is correct
     */
    
    arma::vec logdensity = arma::zeros<arma::vec>(nclusters);
    arma::vec zprob = arma::zeros<arma::vec>(nclusters);
    cluster.AddMarginalContribution(logdensity);
    
    arma::vec logdensity_test = arma::zeros<arma::vec>(nclusters);
    arma::vec zprob_test = arma::zeros<arma::vec>(nclusters);
    // compare with full marginal, a dirichlet-multinomial distribution
    double alpha = cluster.prior_concentration / nclusters;
    double alpha_sum = cluster.prior_concentration;
    for (int k=0; k<nclusters; k++) {
        logdensity_test(k) += lgamma(alpha_sum) - lgamma(alpha_sum + ndata);
        for (int kk=0; kk<nclusters; kk++) {
            double this_zcount = zcounts(kk);
            if (kk == k) {
                this_zcount += 1.0; // add in contribution for zlabel(test_idx) = k
            }
            logdensity_test(k) += lgamma(alpha + this_zcount) - lgamma(alpha);
        }
    }
    zprob = arma::exp(logdensity - logdensity.max()) / arma::sum(arma::exp(logdensity - logdensity.max()));
    zprob_test = arma::exp(logdensity_test - logdensity_test.max()) / arma::sum(arma::exp(logdensity_test - logdensity_test.max()));
    int nequal = 0;
    for (int k=0; k<nclusters; k++) {
        nequal += approximately_equal(zprob(k), zprob_test(k));
    }
    REQUIRE(nequal == nclusters);
    
    /*
     *  now test contribution from categoricals
     */
    std::vector<std::shared_ptr<CategoricalPop> > catobjects = cluster.GetCategoricals();
    
    cluster.AddCategoricalContribution(logdensity, test_categories);
    
    for (int l=0; l<catobjects.size(); l++) {
        arma::mat n_kj = cluster.GetCategoryCounts(l);  // counts have already had test_idx removed
        arma::vec alpha_l = arma::exp(catobjects[l]->Value());
        double alpha_sum = arma::sum(alpha_l);
        for (int k=0; k<nclusters; k++) {
            for (int kk=0; kk<nclusters; kk++) {
                double this_zcounts = zcounts(kk);
                if (kk == k) {
                    this_zcounts += 1;
                }
                logdensity_test(k) += lgamma(alpha_sum) - lgamma(alpha_sum + this_zcounts);
                for (int j=0; j<catobjects[l]->GetNcategories(); j++) {
                    double this_gcounts = n_kj(kk, j);
                    if (kk == k && j == test_categories[l]) {
                        this_gcounts += 1;  // include contribution from test_idx for zlabel(test_idx) = k
                    }
                    logdensity_test(k) += lgamma(alpha_l(j) + this_gcounts) - lgamma(alpha_l(j));
                }
            }
        }
    }
    
    zprob = arma::exp(logdensity - logdensity.max()) / arma::sum(arma::exp(logdensity - logdensity.max()));
    zprob_test = arma::exp(logdensity_test - logdensity_test.max()) / arma::sum(arma::exp(logdensity_test - logdensity_test.max()));
    nequal = 0;
    for (int k=0; k<nclusters; k++) {
        nequal += approximately_equal(zprob(k), zprob_test(k));
    }
    REQUIRE(nequal == nclusters);
    
    /*
     *  now test contribution from bounded count objects
     */
    std::vector<std::shared_ptr<BoundedCountsPop> > bobjects = cluster.GetBoundedCounts();
    arma::uvec zlabels = cluster.Value();

    cluster.AddBoundedContribution(logdensity, zlabels, test_idx);

    for (int l=0; l<bobjects.size(); l++) {
        double alpha_l = exp(bobjects[l]->Value()(0));
        double beta_l = exp(bobjects[l]->Value()(1));
        arma::uvec counts_l = bobjects[l]->GetData();
        for (int k=0; k<nclusters; k++) {
            // switch the zlabel
            zlabels(test_idx) = k;
            // first compute normalization
            for (int i=0; i<ndata; i++) {
                logdensity_test(k) -= log(bobjects[l]->nmax + 1.0) - lbeta(bobjects[l]->nmax - counts_l(i) + 1.0, counts_l(i) + 1.0);
            }
            for (int kk=0; kk<nclusters; kk++) {
                arma::uvec this_cluster = arma::find(zlabels == kk);
                double counts_sum;
                if (this_cluster.n_elem > 0) {
                    counts_sum = arma::sum(counts_l.elem(this_cluster));
                } else {
                    counts_sum = 0.0;
                }
                double barg1 = counts_sum + alpha_l;
                double barg2 = this_cluster.n_elem * bobjects[l]->nmax + beta_l - counts_sum;
                logdensity_test(k) += lbeta(barg1, barg2) - lbeta(alpha_l, beta_l);
            }
        }
    }
    
    zprob = arma::exp(logdensity - logdensity.max()) / arma::sum(arma::exp(logdensity - logdensity.max()));
    zprob_test = arma::exp(logdensity_test - logdensity_test.max()) / arma::sum(arma::exp(logdensity_test - logdensity_test.max()));
    nequal = 0;
    for (int k=0; k<nclusters; k++) {
        nequal += approximately_equal(zprob(k), zprob_test(k));
    }
    REQUIRE(nequal == nclusters);
    
    /*
     *  now test contribution from bounded count objects
     */
    std::vector<std::shared_ptr<UnboundedCountsPop> > ubobjects = cluster.GetUnboundedCounts();
    
    cluster.AddUnboundedContribution(logdensity, zlabels, test_idx);
    
    for (int l=0; l<ubobjects.size(); l++) {
        double alpha_l = exp(ubobjects[l]->Value()(0));
        double beta_l = exp(ubobjects[l]->Value()(1));
        double rfail_l = exp(ubobjects[l]->Value()(2));
        arma::uvec counts_l = ubobjects[l]->GetData();
        for (int k=0; k<nclusters; k++) {
            // switch the zlabel
            zlabels(test_idx) = k;
            // first compute normalization
            for (int i=0; i<ndata; i++) {
                logdensity_test(k) += lgamma(counts_l(i)+ rfail_l) - lgamma(counts_l(i) + 1.0) - lgamma(rfail_l);
            }
            for (int kk=0; kk<nclusters; kk++) {
                arma::uvec this_cluster = arma::find(zlabels == kk);
                double counts_sum;
                if (this_cluster.n_elem > 0) {
                    counts_sum = arma::sum(counts_l.elem(this_cluster));
                } else {
                    counts_sum = 0.0;
                }
                double barg1 = counts_sum + alpha_l;
                double barg2 = this_cluster.n_elem * rfail_l + beta_l;
                logdensity_test(k) += lbeta(barg1, barg2) - lbeta(alpha_l, beta_l);
            }
        }
    }
    
    zprob = arma::exp(logdensity - logdensity.max()) / arma::sum(arma::exp(logdensity - logdensity.max()));
    zprob_test = arma::exp(logdensity_test - logdensity_test.max()) / arma::sum(arma::exp(logdensity_test - logdensity_test.max()));
    nequal = 0;
    for (int k=0; k<nclusters; k++) {
        nequal += approximately_equal(zprob(k), zprob_test(k));
    }

    REQUIRE(nequal == nclusters);
}

// run MCMC sampler with 2 bounded counts objects, 3 categorical objects, 4 clusters. see if we recover the correct proportions among the
// clusters and the correct cluster-dependent markov transition matrices
TEST_CASE("Test Sampler for 2 bounded counts objects, 3 clusters", "[bounded counts]") {
    // generate the test data
    // first generate the data
    int ndata = 10000;
    int nclusters = 4;
    int ncount_pars = 2;
    int ncat_pars = 3;
    int nstates = 4;
    
    arma::vec pi = arma::randu<arma::vec>(nclusters);
    pi /= arma::sum(pi);

    // cluster labels
    arma::uvec zlabels = generate_cluster_labels(ndata, arma::conv_to<std::vector<double> >::from(pi));
    
    // bounded count data
    std::vector<arma::uvec> counts;
    arma::uvec nmax = {10, 60};
    for (int l=0; l<ncount_pars; l++) {
        arma::vec probs = arma::randu<arma::vec>(nclusters);
        counts.push_back(generate_bounded_counts(zlabels, probs, nmax[l]));
    }
    
    // categorical data
    std::vector<arma::uvec> categoricals;
    arma::uvec ncategories = {2, 5, 20};
    for (int l=0; l<ncat_pars; l++) {
        arma::mat probs(nclusters, ncategories[l]);
        for (int k=0; k<nclusters; k++) {
            probs.row(k) = arma::randu<arma::rowvec>(ncategories(l));
            probs.row(k) /= arma::sum(probs.row(k));
        }
        categoricals.push_back(generate_categoricals(zlabels, probs));
    }

    // markov chains
    std::vector<arma::mat> Tmats;
    for (int k=0; k<nclusters; k++) {
        arma::mat Tmat_k = arma::randu<arma::mat>(nstates, nstates);
        // row sums must be normalized to one
        for (int r=0; r<nstates; r++) {
            Tmat_k.row(r) /= arma::sum(Tmat_k.row(r));
        }
        Tmats.push_back(Tmat_k);
    }
    
    std::vector<std::vector<int> > mchains;
    for (int i=0; i<ndata; i++) {
        mchains.push_back(generate_markov_chain(1, Tmats[zlabels(i)])[0]);
    }
    
    // instantiate the parameter objects
    
    // cluster labels first
    std::shared_ptr<ClusterLabels> Cluster = std::make_shared<ClusterLabels>(true, "Z", ndata, nclusters);
    
    // now markov chain parameters
    std::vector<std::shared_ptr<TransitionProbability> > Tprobs;
    for (int k=0; k<nclusters; k++) {
        std::string pname("Tprob-");
        pname += std::to_string(k);
        Tprobs.push_back(std::make_shared<TransitionProbability>(true, pname, mchains, nstates, k));
    }
    
    std::vector<std::shared_ptr<TransitionPopulation> > Gammas;
    for (int r=0; r<nstates; r++) {
        for (int c=0; c<nstates; c++) {
            std::string pname("gamma-");
            pname += std::to_string(r);
            pname += "-";
            pname += std::to_string(c);
            Gammas.push_back(std::make_shared<TransitionPopulation>(false, pname, r, c));
        }
    }
    
    std::vector<std::shared_ptr<TransitionHyperPrior> > Hyper;
    for (int i=0; i<nstates + 1; i++) {
        std::string pname("hyper-");
        if (i < nstates) {
            pname += std::to_string(i);
        } else {
            pname += "diag";
        }
        Hyper.push_back(std::make_shared<TransitionHyperPrior>(false, pname));
    }
    
    // bounded counts objects
    std::vector<std::shared_ptr<BoundedCountsPop> > Bcounts;
    for (int l=0; l<ncount_pars; l++) {
        std::string pname("bcounts-");
        pname += std::to_string(l);
        Bcounts.push_back(std::make_shared<BoundedCountsPop>(false, pname, counts[l], nmax[l]));
    }
    
    // categorical objects
    std::vector<std::shared_ptr<CategoricalPop> > Cats;
    for (int l=0; l<ncat_pars; l++) {
        std::string pname("categorical-");
        pname += std::to_string(l);
        Cats.push_back(std::make_shared<CategoricalPop>(false, pname, categoricals[l]));
    }
    
    /*
     *  connect the parameter objects
     */
    
    // connect the transition matrices for each cluster
    for (int k=0; k<nclusters; k++) {
        Cluster->AddTransitionMatrix(Tprobs[k]);
        Tprobs[k]->SetClusterLabels(Cluster);
        for (int l=0; l<Gammas.size(); l++) {
            Tprobs[k]->AddPopulationPtr(Gammas[l]);
            Gammas[l]->AddTransitionMatrix(Tprobs[k]);
        }
    }
    
    // connect the gammas
    for (int l=0; l<Gammas.size(); l++) {
        int row = Gammas[l]->row_idx;
        for (int m=0; m<Gammas.size(); m++) {
            if (Gammas[m]->row_idx == row && m != l) {
                Gammas[l]->AddGamma(Gammas[m]);
            }
        }
    }
    
    // connect the gammas and their prior
    for (int l=0; l<Gammas.size(); l++) {
        int row = Gammas[l]->row_idx;
        int col = Gammas[l]->col_idx;
        if (row == col) {
            // give gammas along the diagonal their own prior
            Hyper.back()->AddTransitionPop(Gammas[l]);
            Gammas[l]->SetHyperPrior(Hyper.back());
        } else {
            // give all gammas along this column their own prior
            Hyper[col]->AddTransitionPop(Gammas[l]);
            Gammas[l]->SetHyperPrior(Hyper[col]);
        }
    }
    
    // connect the bounded count and categoricals
    for (int l=0; l<ncount_pars; l++) {
        Bcounts[l]->SetClusterLabels(Cluster);
        Cluster->AddBoundedCountsPop(Bcounts[l]);
    }
    for (int l=0; l<ncat_pars; l++) {
        Cats[l]->SetClusterLabels(Cluster);
        Cluster->AddCategoricalPop(Cats[l]);
    }
    
    
}