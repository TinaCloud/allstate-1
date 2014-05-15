//
//  cluster.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include "cluster.hpp"

ClusterLabels::ClusterLabels(bool track, std::string label, int ndata, int nclusters, double prior_concentration,
                             double temperature) : Parameter<arma::uvec>(track, label, temperature), ndata_(ndata),
                                nclusters_(nclusters), prior_concentration_(prior_concentration)
{
    value_.set_size(ndata);
    cluster_counts_.set_size(nclusters_);
}

// just set random cluster labels from a uniform distribution. also sets the starting value.
arma::uvec ClusterLabels::StartingValue()
{
    value_ = arma::randi<arma::uvec>(ndata_, arma::distr_param(0, nclusters_-1));
    CountCategories();
    CountClusters();
    return value_;
}

void ClusterLabels::CountClusters()
{
    for (int i=0; i<ndata_; i++) {
        cluster_counts_(value_(i))++;
    }
}

void ClusterLabels::CountCategories()
{
    for (int l=0; l<categoricals_.size(); l++) {
        arma::umat n_kj(nclusters_, categoricals_[l]->GetNcategories());
        arma::uvec categories_l = categoricals_[l]->GetData();
        for (int i=0; i<ndata_; i++) {
            n_kj(value_(i), categories_l(i)-1) += 1;
        }
        category_counts_[l] = n_kj;
    }
}

arma::uvec ClusterLabels::RandomPosterior()
{
    arma::uvec zvalues = value_;
    // compute probability of cluster label given others, one-at-a-time
    for (int i=0; i<ndata_; i++) {
        arma::vec log_zprob(nclusters_);
        
        // first update the cluster and category counts after removing this data point
        int this_cluster = value_(i);
        cluster_counts_(this_cluster)--;
        std::vector<int> this_category(categoricals_.size());
        for (int l=0; l<categoricals_.size(); l++) {
            // grab the l-th categorical data point value for the i-th data point
            this_category[l] = categoricals_[l]->GetData()(i);
            category_counts_[l](this_cluster, this_category[l])--;
        }
        
        // now get conditional log-posteriors
        for (int k=0; k<nclusters_; k++) {
            // start with marginal probability for this cluster label
            log_zprob(k) = log(cluster_counts_(k) + prior_concentration_ / nclusters_);
            
            // now add in contributions from categoricals
            for (int l=0; l<categoricals_.size(); l++) {
                log_zprob(k) += log(category_counts_[l](k, this_category[l]) + categoricals_[l]->Value()(this_category[l])) -
                    log(cluster_counts_(k) + arma::sum(categoricals_[l]->Value()));
            }
            
            // add in contributions from bounded count objects
            
            
            
            // add in contributions from unbounded count objects
            
            
            
            // TODO: add in contribution from Markov chains for each cluster
        }
    }
    return zvalues;
}