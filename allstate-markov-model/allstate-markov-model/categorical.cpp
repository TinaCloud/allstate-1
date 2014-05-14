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
CategoricalPop::CategoricalPop(bool track, std::string label, arma::uvec data, double temperature, double prior_shape,
                                   double prior_scale) : Parameter::Parameter(track, label, temperature), data_(data), prior_scale_(prior_scale), prior_shape_(prior_shape)
{
    ndata_ = data_.n_cols;
    ncategories_ = data_.max();
    value_.resize(ncategories_);
    // make sure categories have values j = 1, 2, ..., ncategories
    arma::uvec ucats = arma::unique(data);
    for (int j=1; j<=ncategories_; j++) {
        assert(ucats(j-1) = j);
    }
}

// set the starting value by just drawing from the prior
void CategoricalPop::SetStartingValue() {
    for (int j=0; j<ncategories_; j++) {
        value_(j) = RandGen.gamma(prior_shape_, prior_scale_);
    }
}

// compute the conditional log-posterior of the population parameter of this categorical variable
double CategoricalPop::LogDensity(arma::vec alpha) {
    int nclusters = cluster_labels_->GetNclusters();
    double alpha_sum = arma::sum(alpha);
    // first count the number of times category j is in cluster k
    arma::umat n_jk(nclusters, ncategories_);
    arma::uvec zvalues = cluster_labels_->Value();
    for (int i=0; i<ndata_; i++) {
        n_jk(zvalues(i), data_[i]-1) += 1;
    }
    arma::uvec n_k = arma::sum(n_jk, 1);
    double logdensity = nclusters * lgamma(alpha_sum) + (prior_shape_ - 1.0) * arma::sum(arma::log(alpha)) - alpha_sum / prior_scale_;
    for (int j=0; j<ncategories_; j++) {
        logdensity -= nclusters * lgamma(alpha(j));
        for (int k=0; k<nclusters; k++) {
            logdensity += lgamma(n_jk(k,j) + alpha(j));
        }
    }
    for (int k=0; k<nclusters; k++) {
        logdensity += lgamma(alpha_sum + n_k(k));
    }
    
    return logdensity;
}