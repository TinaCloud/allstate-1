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
#include "markov_chain.hpp"

// return the logarithm of the Beta function
double lbeta(double x, double y);

class ClusterLabels : public Parameter<arma::uvec> {
    arma::vec cluster_counts_;  // number of data points occupying each cluster
    std::vector<arma::mat> category_counts_;  // number of data points occupying each cluster and category for each cateogorical
    
    // pointers to parameter objects, either a nparameters vector
    std::vector<std::shared_ptr<CategoricalPop> > categoricals_; // list of categorical parameters
    std::vector<std::shared_ptr<BoundedCountsPop> > bounded_counts_; // list of count parameters with upper bound
    std::vector<std::shared_ptr<UnboundedCountsPop> > unbounded_counts_; // count parameter with no upper bound, one per cluster
    std::vector<std::shared_ptr<TransitionProbability> > transition_matrices_; // markov transition matrix parameter, one per cluster
    
    std::vector<std::vector<int> >& markov_chain_;
    
public:
    int ndata;  // number of data points
    int nclusters;  // number of clusters
    double prior_concentration;  // the concentration parameter for the symmetric dirichlet prior on the cluster weights

    ClusterLabels(bool track, std::string label, int nclusters, std::vector<std::vector<int> >& markov_chain, double prior_concentration=1.0, double temperature=1.0);
    
    arma::uvec RandomPosterior();
    
    // add in the contribution to the conditional log-posterior from p(Z_i=k|Z_j, j \neq i)
    void AddMarginalContribution(arma::vec& log_zprob);
    
    // add in the contribution to the conditional log-posterior from the categorical data
    void AddCategoricalContribution(arma::vec& log_zprob, std::vector<int>& category);
    
    // add in the contribution to the conditional log-posterior from the bounded count data
    void AddBoundedContribution(arma::vec& log_zprob, arma::uvec& zvalues, int i);
    
    // add in the contribution to the conditional log-posterior from the unbounded count data
    void AddUnboundedContribution(arma::vec& log_zprob, arma::uvec& zvalues, int i);
    
    // add in the contribution to the conditional log-posterior from the Markov chain data
    void AddMarkovContribution(arma::vec& log_zprob, int data_id);
    
    // set and return the starting value
    arma::uvec StartingValue();
    
    // count the number of data points in each cluster and save internally
    void CountClusters();
    
    // count the number of data point in each cluster with each unique category value and save internally
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
    
    void AddTransitionMatrix(std::shared_ptr<TransitionProbability> new_matrix) {
        transition_matrices_.push_back(new_matrix);
    }
    
    // return the number of data points in each cluster
    arma::vec GetClusterCounts() {
        return cluster_counts_;
    }
    
    // get the category counts (nclusters, ncategories) for categorical parameters object of index idx
    arma::mat GetCategoryCounts(int idx) {
        return category_counts_[idx];
    }
    
    std::vector<std::shared_ptr<CategoricalPop> > GetCategoricals() { return categoricals_; }
    std::vector<std::shared_ptr<BoundedCountsPop> > GetBoundedCounts() { return bounded_counts_; }
    std::vector<std::shared_ptr<UnboundedCountsPop> > GetUnboundedCounts() { return unbounded_counts_; }
    std::vector<std::shared_ptr<TransitionProbability> > GetTransitionMatrices() { return transition_matrices_; }
    
    
    // recalculate the cluster and category counts after removing the indexed cluster label
    std::vector<int> RemoveClusterLabel(unsigned int idx);
    
    // update the cluster and category counts with a new cluster label
    void UpdateClusterCounts(unsigned int idx, unsigned int cluster_idx);
};


#endif /* defined(__allstate_markov_model__cluster__) */
