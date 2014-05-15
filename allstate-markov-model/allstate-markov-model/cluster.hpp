//
//  cluster.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__cluster__
#define __allstate_markov_model__cluster__

// standard includes
#include <iostream>
#include <vector>
#include <memory>

// local includes
#include "categorical.hpp"
#include "bounded_counts.hpp"
#include "unbounded_counts.hpp"

class ClusterLabels : public Parameter<arma::uvec> {
    int ndata_;  // number of data points
    int nclusters_;  // number of clusters
    double prior_concentration_;  // the concentration parameter for the symmetric dirichlet prior on the cluster weights
    arma::uvec cluster_counts_;  // number of data points occupying each cluster
    std::vector<arma::umat> category_counts_;  // number of data points occupying each cluster and category for each cateogorical
    
    // pointers to parameter objects, either a nparameters vector
    std::vector<std::shared_ptr<CategoricalPop> > categoricals_; // list of categorical parameters
    std::vector<std::shared_ptr<BoundedCountsPop> > bounded_counts_; // list of count parameters with upper bound
    std::vector<std::shared_ptr<UnboundedCountsPop> > unbounded_counts_; // count parameter with no upper bound, one per cluster
    // std::vector<std::shared_ptr<TransitionMatrix> > transition_matrix_; // markov transition matrix parameter, one per cluster
    
public:
    ClusterLabels(bool track, std::string label, int ndata, int nclusters, double prior_concentration=1.0, double temperature=1.0);
    
    arma::uvec RandomPosterior();
    
    arma::uvec StartingValue();
    
    int GetNclusters() { return nclusters_; }
    
    void CountClusters();
    void CountCategories();
    
    // routines to add new population-level parameter objects to the parameter list.
    void AddCategoricalPop(std::shared_ptr<CategoricalPop> new_categorical) {
        categoricals_.push_back(new_categorical);
        category_counts_.resize(categoricals_.size());
        categoricals_.back()->SetIndex(categoricals_.size()-1);
    }
    void AddBoundedCountsPop(std::shared_ptr<BoundedCountsPop> new_bounded_counts) {
        bounded_counts_.push_back(new_bounded_counts);
    }
    void AddUnboundedCountsPop(std::shared_ptr<UnboundedCountsPop> new_unbounded_counts) {
        unbounded_counts_.push_back(new_unbounded_counts);
    }
    
    arma::uvec GetClusterCounts() {
        return cluster_counts_;
    }
    
    // get the category counts (nclusters, ncategories) for categorical parameters object of index idx
    arma::umat GetCategoryCounts(int idx) {
        return category_counts_[idx];
    }
};


#endif /* defined(__allstate_markov_model__cluster__) */
