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
    double prior_shape_; // shape parameter for gamma prior
    double prior_scale_; // scale parameter for gamma prior. prior expectation is prior_shape_ * prior_scale_
    int ndata_; // the number of data points
    int ncategories_; // the number of unique observed categories, assumes category labels are 1, 2, ..., ncategories
    arma::uvec data_; // the observed categories, a vector with ndata elements containing ncategories possible values
    std::shared_ptr<ClusterLabels> cluster_labels_; // labels specifying which cluster each data point belongs to
    
public:
    CategoricalPop(bool track, std::string label, arma::uvec data, double temperature=1.0, double prior_shape=2.0,
                     double prior_scale=0.5);
    
    double LogDensity(arma::vec alpha);
    
    void SetStartingValue();
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels> labels) {
        cluster_labels_ = labels;
    }
    
    arma::uvec GetData() {
        return data_;
    }
};


#endif /* defined(__allstate_markov_model__categorical__) */
