//
//  unbounded_counts.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__unbounded_counts__
#define __allstate_markov_model__unbounded_counts__

#include <iostream>
#include <memory>

// local includes
#include <parameters.hpp>
#include <armadillo>

class ClusterLabels;

class UnboundedCountsPop : public Parameter<arma::vec> {
    arma::uvec& data_; // the observed counts, a vector with ndata elements containing values 0, ..., nmax.
    std::shared_ptr<ClusterLabels> cluster_labels_; // labels specifying which cluster each data point belongs to
    
public:
    double prior_ashape; // shape parameter for gamma prior on beta function parameters
    double prior_ascale; // scale parameter for gamma prior. prior expectation is prior_shape_ * prior_scale_
    double prior_rshape; // shape parameter for gamma prior on number of failures, r
    double prior_rscale;
    int ndata; // the number of data points

    UnboundedCountsPop(bool track, std::string label, arma::uvec& data, double temperature=1.0, double prior_ashape=2.0,
                       double prior_ascale=0.5, double prior_rshape=2.0, double prior_rscale=20.0);
    
    double LogDensity(arma::vec alpha);
    
    arma::vec StartingValue();
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels> labels) {
        cluster_labels_ = labels;
    }
    
    std::shared_ptr<ClusterLabels> GetClusterLabels() { return cluster_labels_; }
    
    arma::uvec GetData() {
        return data_;
    }
    
    
};



#endif /* defined(__allstate_markov_model__unbounded_counts__) */
