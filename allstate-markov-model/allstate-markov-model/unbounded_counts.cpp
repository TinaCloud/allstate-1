//
//  unbounded_counts.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

// boost
#include <boost/math/special_functions/gamma.hpp>

// local includes
#include "unbounded_counts.hpp"
#include "cluster.hpp"

using boost::math::lgamma;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// constructor
UnboundedCountsPop::UnboundedCountsPop(bool track, std::string label, arma::uvec& data, double temperature,
                                       double prior_ashape, double prior_ascale, double prior_rshape, double prior_rscale) :
                                        Parameter<arma::vec>(track, label, temperature), data_(data), prior_ascale_(prior_ascale),
                                        prior_ashape_(prior_ashape), prior_rscale_(prior_rscale), prior_rshape_(prior_rshape)
{
    ndata_ = data_.n_cols;
    value_.resize(3);
}

// set the starting value by just drawing from the prior
arma::vec UnboundedCountsPop::StartingValue()
{
    arma::vec theta(3);
    theta(0) = RandGen.gamma(prior_ashape_, prior_ascale_);
    theta(1) = RandGen.gamma(prior_ashape_, prior_ascale_);
    theta(2) = RandGen.gamma(prior_rshape_, prior_rscale_);
    return arma::log(theta); // convert to log-scale for sampling
}

// compute the conditional log-posterior of the population parameter of this bounded counts variable
double UnboundedCountsPop::LogDensity(arma::vec alpha)
{
    alpha = arma::exp(alpha); // sampling is done on log scale, but computations are done on original scale
    int nclusters = cluster_labels_->GetNclusters();
    arma::uvec zvalues = cluster_labels_->Value();

    // start with contribution from prior
    double logdensity = (prior_ashape_ - 1.0) * (log(alpha(0) + log(alpha(1)))) - (alpha(0) + alpha(1)) / prior_ascale_ +
        (prior_rshape_ - 1.0) * log(alpha(2)) - alpha(2) / prior_rscale_;
    
    // get contribution from data
    logdensity -= nclusters * (lgamma(alpha(0)) + lgamma(alpha(1)) - lgamma(alpha(0) + alpha(1))) - ndata_ * lgamma(alpha(2));
    for (int k=0; k<nclusters; k++) {
        // first get total counts for this cluster
        arma::uvec cluster_idx = arma::find(zvalues == k);
        unsigned int n_k = cluster_idx.n_cols;  // total number of data points in this cluster
        unsigned int counts_sum = arma::sum(data_.elem(cluster_idx));
        logdensity += lgamma(alpha(0) + counts_sum) + lgamma(alpha(1) + n_k * alpha(2)) -
            lgamma(alpha(0) + alpha(1) + n_k * alpha(2) + counts_sum);
        for (int i=0; i<n_k; i++) {
            logdensity += lgamma(data_(cluster_idx(i)) + alpha(2)) - lgamma(data_(cluster_idx(i)) + 1.0);
        }
    }
    
    return logdensity;
}