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
                                       double ashape, double ascale, double rshape, double rscale) :
                                        Parameter<arma::vec>(track, label, temperature), data_(data), prior_ascale(ascale),
                                        prior_ashape(ashape), prior_rscale(rscale), prior_rshape(rshape)
{
    ndata = data_.n_elem;
    value_.resize(3);
    value_.zeros();
}

// set the starting value by just drawing from the prior
arma::vec UnboundedCountsPop::StartingValue()
{
    arma::vec theta(3);
    theta(0) = RandGen.gamma(prior_ashape, prior_ascale);
    theta(1) = RandGen.gamma(prior_ashape, prior_ascale);
    theta(2) = RandGen.gamma(prior_rshape, prior_rscale);
    return arma::log(theta); // convert to log-scale for sampling
}

// compute the conditional log-posterior of the population parameter of this bounded counts variable
double UnboundedCountsPop::LogDensity(arma::vec log_alpha)
{
    arma::vec alpha = arma::exp(log_alpha); // sampling is done on log scale, but computations are done on original scale
    int nclusters = cluster_labels_->nclusters;
    arma::uvec zvalues = cluster_labels_->Value();

    // start with contribution from prior
    double logdensity = (prior_ashape - 1.0) * (log_alpha(0) + log_alpha(1)) - (alpha(0) + alpha(1)) / prior_ascale +
        (prior_rshape - 1.0) * log_alpha(2) - alpha(2) / prior_rscale;
    
    // get contribution from data
    logdensity += -nclusters * lbeta(alpha(0), alpha(1)) - ndata * lgamma(alpha(2));
    
    arma::vec zcounts = cluster_labels_->GetClusterCounts(); // total number of data points in each cluster
    arma::vec counts_sum = arma::zeros<arma::vec>(nclusters);

    // first get total counts for each cluster
    for (int i=0; i<ndata; i++) {
        counts_sum(zvalues(i)) += data_(i);
    }
    
    for (int k=0; k<nclusters; k++) {
        logdensity += lbeta(alpha(0) + counts_sum(k), alpha(1) + zcounts(k) * alpha(2));
    }
    // add contribution from normalization
    for (int i=0; i<ndata; i++) {
        logdensity += lgamma(data_(i) + alpha(2));
    }
    
    return logdensity;
}