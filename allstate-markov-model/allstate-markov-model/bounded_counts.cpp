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
BoundedCountsPop::BoundedCountsPop(bool track, std::string label, arma::uvec data, int nmax, double temperature, double prior_shape,
                               double prior_scale) : Parameter::Parameter(track, label, temperature), data_(data), prior_scale_(prior_scale), prior_shape_(prior_shape), nmax_(nmax)
{
    ndata_ = data_.n_cols;
    value_.resize(2);
}

// set the starting value by just drawing from the prior
void BoundedCountsPop::SetStartingValue() {
    value_(0) = RandGen.gamma(prior_shape_, prior_scale_);
    value_(1) = RandGen.gamma(prior_shape_, prior_scale_);
}

// compute the conditional log-posterior of the population parameter of this bounded counts variable
double BoundedCountsPop::LogDensity(arma::vec alpha) {
    int nclusters = cluster_labels_->GetNclusters();
    arma::uvec zvalues = cluster_labels_->Value();
    // start with contribution from prior
    double logdensity = (prior_shape_ - 1.0) * (log(alpha(0) + log(alpha(1)))) - (alpha(0) + alpha(1)) / prior_scale_;
    // get contribution from data
    logdensity -= nclusters * (lgamma(alpha(0)) + lgamma(alpha(1)) - lgamma(alpha(0) + alpha(1)));
    for (int k=0; k<nclusters; k++) {
        // first get total counts for this cluster
        arma::uvec cluster_idx = arma::find(zvalues == k);
        unsigned int n_k = cluster_idx.n_cols;  // total number of data points in this cluster
        unsigned int counts_sum = arma::sum(data_.elem(cluster_idx));
        logdensity += lgamma(alpha(0) + counts_sum) + lgamma(alpha(1) + n_k * nmax_ - counts_sum) -
            lgamma(n_k * nmax_ + alpha(0) + alpha(1));
    }
    
    return logdensity;
}