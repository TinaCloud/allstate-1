//
//  categorical.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__categorical__
#define __allstate_markov_model__categorical__

// standard includes
#include <iostream>
#include <memory>

// local includes
#include <armadillo>
#include <parameters.hpp>

class ClusterLabels;

/*
 * CLASS FOR THE POPULATION-LEVEL PARAMETERS OF A CATEGORICAL VARIABLE
 */

class CategoricalPop : public Parameter<arma::vec>
{
    int ncategories_; // the number of unique observed categories, assumes category labels are 1, 2, ..., ncategories
    arma::uvec& data_; // the observed categories, a vector with ndata elements containing ncategories possible values
    std::shared_ptr<ClusterLabels>& cluster_labels_; // labels specifying which cluster each data point belongs to
    int idx_; // the index in the stack of categoricals in cluster_labels_
    
public:
    double prior_shape; // shape parameter for gamma prior
    double prior_scale; // scale parameter for gamma prior. prior expectation is prior_shape_ * prior_scale_
    int ndata; // the number of data points

    CategoricalPop(bool track, std::string label, arma::uvec& data, double temperature=1.0, double prior_shape=2.0,
                     double prior_scale=0.5);
    
    double LogDensity(arma::vec alpha);
    
    arma::vec StartingValue();
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels>& labels) {
        cluster_labels_ = labels;
    }
    
    void SetIndex(int idx) {
        idx_ = idx;
    }
    
    std::shared_ptr<ClusterLabels> GetClusterLabels() { return cluster_labels_; }
    
    arma::uvec GetData() {
        return data_;
    }
    
    int GetNcategories() {
        return ncategories_;
    }
    
    int GetIndex() {
        return idx_;
    }
};


#endif /* defined(__allstate_markov_model__categorical__) */
