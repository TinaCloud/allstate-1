//
//  categorical.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include <cassert>

// boost includes
#include <boost/math/special_functions/gamma.hpp>

// local includes
#include <random.hpp>
#include "categorical.hpp"
#include "cluster.hpp"

using boost::math::lgamma;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// constructor
CategoricalPop::CategoricalPop(bool track, std::string label, arma::uvec& data, double temperature, double prior_shape,
                               double prior_scale) : Parameter<arma::vec>(track, label, temperature), data_(data), prior_scale_(prior_scale), prior_shape_(prior_shape)
{
    ndata_ = data_.n_cols;
    ncategories_ = data_.max() + 1;
    value_.resize(ncategories_);
    // make sure categories have values j = 0, 2, ..., ncategories - 1
    arma::uvec ucats = arma::unique(data);
    for (int j=0; j<ncategories_; j++) {
        assert(ucats(j) = j);
    }
}

// set the starting value by just drawing from the prior
arma::vec CategoricalPop::StartingValue()
{
    arma::vec alpha(ncategories_);
    for (int j=0; j<ncategories_; j++) {
        alpha(j) = RandGen.gamma(prior_shape_, prior_scale_);
    }
    return arma::log(alpha);  // run sampler on log scale
}

// compute the conditional log-posterior of the population parameter of this categorical variable
double CategoricalPop::LogDensity(arma::vec alpha)
{
    alpha = arma::exp(alpha);  // sampling is done on log scale, so convert back to linear scale
    int nclusters = cluster_labels_->GetNclusters();
    double alpha_sum = arma::sum(alpha);
    
    // grab the counts for the number of times category j is in cluster k, and the number of data points in cluster k
    arma::umat n_kj = cluster_labels_->GetCategoryCounts(idx_);
    arma::uvec n_k = cluster_labels_->GetClusterCounts();
    
    // compute log-posterior
    double logdensity = (prior_shape_ - 1.0) * arma::sum(arma::log(alpha)) - alpha_sum / prior_scale_;  // log-prior
    logdensity += nclusters * lgamma(alpha_sum);  // terms in log-likelihood
    for (int j=0; j<ncategories_; j++) {
        logdensity -= nclusters * lgamma(alpha(j));
        for (int k=0; k<nclusters; k++) {
            logdensity += lgamma(n_kj(k,j) + alpha(j));
        }
    }
    for (int k=0; k<nclusters; k++) {
        logdensity += lgamma(alpha_sum + n_k(k));
    }
    
    return logdensity;
}