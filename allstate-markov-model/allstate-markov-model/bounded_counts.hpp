//
//  bounded_counts.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__bounded_counts__
#define __allstate_markov_model__bounded_counts__

#include <iostream>
#include <memory>

// local includes
#include <parameters.hpp>
#include <armadillo>

class ClusterLabels;

class BoundedCountsPop : public Parameter<arma::vec> {
    arma::uvec& data_; // the observed counts, a vector with ndata elements containing values 0, ..., nmax.
    std::shared_ptr<ClusterLabels>& cluster_labels_; // labels specifying which cluster each data point belongs to
    
public:
    double prior_shape; // shape parameter for gamma prior
    double prior_scale; // scale parameter for gamma prior. prior expectation is prior_shape_ * prior_scale_
    int ndata; // the number of data points
    int nmax; // the maximum number of counts
    
    BoundedCountsPop(bool track, std::string label, arma::uvec& data, int nmax, double temperature=1.0, double prior_shape=2.0,
                     double prior_scale=0.5);
    
    double LogDensity(arma::vec alpha);

    arma::vec StartingValue();
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels>& labels) {
        cluster_labels_ = labels;
    }
    
    std::shared_ptr<ClusterLabels> GetClusterLabels() { return cluster_labels_; }
    
    arma::uvec& GetData() {
        return data_;
    }
  
    
};


#endif /* defined(__allstate_markov_model__bounded_counts__) */
