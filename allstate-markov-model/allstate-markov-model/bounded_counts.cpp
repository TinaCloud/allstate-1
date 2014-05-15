//
//  bounded_counts.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

// boost
#include <boost/math/special_functions/gamma.hpp>

// local includes
#include "bounded_counts.hpp"
#include "cluster.hpp"

using boost::math::lgamma;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// constructor
BoundedCountsPop::BoundedCountsPop(bool track, std::string label, arma::uvec& data, int n, double temperature, double prior_sh,
                                   double prior_sc) : Parameter<arma::vec>(track, label, temperature), data_(data),
                                    prior_scale(prior_sc), prior_shape(prior_sh), nmax(n)
{
    ndata = data_.n_cols;
    value_.resize(2);
}

// set the starting value by just drawing from the prior
arma::vec BoundedCountsPop::StartingValue()
{
    arma::vec alpha(2);
    alpha(0) = RandGen.gamma(prior_shape, prior_scale);
    alpha(1) = RandGen.gamma(prior_shape, prior_scale);
    return arma::log(alpha);  // run MCMC sampler on the log scale
}

// compute the conditional log-posterior of the population parameter of this bounded counts variable
double BoundedCountsPop::LogDensity(arma::vec alpha)
{
    alpha = arma::exp(alpha);  // sampling is done on log scale, so convert back to linear scale
    int nclusters = cluster_labels_->GetNclusters();
    arma::uvec zvalues = cluster_labels_->Value();
    // start with contribution from prior
    double logdensity = (prior_shape - 1.0) * (log(alpha(0) + log(alpha(1)))) - (alpha(0) + alpha(1)) / prior_scale;
    // get contribution from data
    logdensity -= nclusters * (lgamma(alpha(0)) + lgamma(alpha(1)) - lgamma(alpha(0) + alpha(1)));
    arma::uvec n_k = cluster_labels_->GetClusterCounts();  // total number of data points in each cluster
    for (int k=0; k<nclusters; k++) {
        // first get total counts for this cluster
        arma::uvec cluster_idx = arma::find(zvalues == k);
        double counts_sum = arma::sum(data_.elem(cluster_idx));
        logdensity += lgamma(alpha(0) + counts_sum) + lgamma(alpha(1) + n_k(k) * nmax - counts_sum) -
            lgamma(n_k(k) * nmax + alpha(0) + alpha(1));
    }
    
    return logdensity;
}