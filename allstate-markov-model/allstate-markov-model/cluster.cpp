//
//  cluster.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/14/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include "cluster.hpp"
#include <boost/random/discrete_distribution.hpp>

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;


// return the logarithm of the Beta function
double lbeta(double x, double y) {
    double lbeta = lgamma(x) + lgamma(y) - lgamma(x + y);
    return lbeta;
}

ClusterLabels::ClusterLabels(bool track, std::string label, int n, int K, double concentration,
                             double temperature) : Parameter<arma::uvec>(track, label, temperature), ndata(n),
                                nclusters(K), prior_concentration(concentration)
{
    value_.set_size(ndata);
    cluster_counts_.set_size(nclusters);
}

// just set random cluster labels from a uniform distribution. also sets the starting value.
arma::uvec ClusterLabels::StartingValue()
{
    value_ = arma::randi<arma::uvec>(ndata, arma::distr_param(0, nclusters-1));
    CountCategories();
    CountClusters();
    return value_;
}

void ClusterLabels::CountClusters()
{
    for (int i=0; i<ndata; i++) {
        cluster_counts_(value_(i))++;
    }
}

void ClusterLabels::CountCategories()
{
    for (int l=0; l<categoricals_.size(); l++) {
        arma::umat n_kj(nclusters, categoricals_[l]->GetNcategories());
        arma::uvec categories_l = categoricals_[l]->GetData();
        for (int i=0; i<ndata; i++) {
            n_kj(value_(i), categories_l(i)-1) += 1;
        }
        category_counts_[l] = n_kj;
    }
}

// update the cluster and category counts after removing the cluster label with input index.
std::vector<int> ClusterLabels::RemoveClusterLabel(unsigned int idx)
{
    int this_cluster = value_(idx);
    cluster_counts_(this_cluster)--;
    std::vector<int> this_category(categoricals_.size());
    for (int l=0; l<categoricals_.size(); l++) {
        // grab the l-th categorical data point value for the i-th data point
        this_category[l] = categoricals_[l]->GetData()(idx);
        category_counts_[l](this_cluster, this_category[l])--;
    }
    return this_category;
}

// increase the counts by one at the specified input indices
void ClusterLabels::UpdateClusterCounts(unsigned int idx, unsigned int cluster_idx)
{
    cluster_counts_[cluster_idx]++;
    for (int l=0; l<categoricals_.size(); l++) {
        int this_category = categoricals_[l]->GetData()(idx);
        category_counts_[l](cluster_idx, this_category)++;
    }
}

// add in the contribution to the conditional log-posterior from p(Z_i=k|Z_j, j \neq i). note that this is done in-place.
void ClusterLabels::AddMarginalContribution(arma::vec& log_zprob)
{
    for (int k=0; k<nclusters; k++) {
        log_zprob(k) += log(cluster_counts_(k) + prior_concentration / nclusters);
    }
}

// add in the contribution to the conditional log-posterior from the categorical data. this is done in-place.
void ClusterLabels::AddCategoricalContribution(arma::vec& log_zprob, std::vector<int>& category)
{
    for (int k=0; k<nclusters; k++) {
        for (int l=0; l<categoricals_.size(); l++) {
            arma::vec alpha_l = arma::exp(categoricals_[l]->Value());  // sampling values on the log scale
            log_zprob(k) += log(category_counts_[l](k, category[l]) + alpha_l(category[l])) -
                log(cluster_counts_(k) + arma::sum(alpha_l));
        }
    }
}

// add in the contribution to the conditional log-posterior from the bounded count data. this is done in-place.
void ClusterLabels::AddBoundedContribution(arma::vec& log_zprob, arma::uvec& zvalues, int i)
{
    for (int l=0; l<bounded_counts_.size(); l++) {
        // first compute the log-beta functions for each cluster after removing this data point. do this here to avoid
        // duplicating the calculations below.
        std::vector<double> counts_sum(nclusters);
        std::vector<double> logbeta(nclusters);
        arma::uvec counts_l = bounded_counts_[l]->GetData();
        arma::vec alpha = arma::exp(bounded_counts_[l]->Value());  // population-level parameters
        for (int k=0; k<nclusters; k++) {
            arma::uvec cluster_idx = arma::find(zvalues == k);  // find data points in this cluster
            counts_sum[k] = arma::sum(counts_l.elem(cluster_idx));
            if (zvalues(i) == k) {
                counts_sum[k] -= counts_l(i);
            }
            logbeta[k] = lbeta(alpha(0) + counts_sum[k],
                               alpha(1) + cluster_counts_(k) * bounded_counts_[l]->nmax - counts_sum[k]);
        }
        double logbeta_sum = std::accumulate(logbeta.begin(), logbeta.end(), 0.0);
        // now calculate log_zprob by adding in the i-th data point assuming z[i] = k one at a time so we don't have to
        // redo the beta function calculations
        for (int k=0; k<nclusters; k++) {
            double barg1 = alpha(0) + counts_sum[k] + counts_l(i);
            double barg2 = (cluster_counts_(k) + 1.0) * bounded_counts_[l]->nmax + alpha(1) - counts_sum[k] - counts_l(i);
            double this_logbeta = lbeta(barg1, barg2);
            log_zprob(k) += logbeta_sum - logbeta[k] + this_logbeta;
        }
    }
}

// add in the contribution to the conditional log-posterior from the unbounded count data. this is done in-place.
void ClusterLabels::AddUnboundedContribution(arma::vec& log_zprob, arma::uvec& zvalues, int i)
{
    for (int l=0; l<unbounded_counts_.size(); l++) {
        // first compute the log-beta functions for each cluster after removing this data point. do this here to avoid
        // duplicating the calculations below.
        std::vector<double> counts_sum(nclusters);
        std::vector<double> logbeta(nclusters);
        arma::uvec counts_l = unbounded_counts_[l]->GetData();
        arma::vec alpha = arma::exp(unbounded_counts_[l]->Value());  // population-level parameters
        for (int k=0; k<nclusters; k++) {
            arma::uvec cluster_idx = arma::find(zvalues == k);  // find data points in this cluster
            counts_sum[k] = arma::sum(counts_l.elem(cluster_idx));
            if (zvalues(i) == k) {
                counts_sum[k] -= counts_l(i);
            }
            logbeta[k] = lbeta(alpha(0) + counts_sum[k], alpha(1) + cluster_counts_(k) * alpha(2));
        }
        double logbeta_sum = std::accumulate(logbeta.begin(), logbeta.end(), 0.0);
        // now calculate log_zprob by adding in the i-th data point assuming z[i] = k one at a time so we don't have to
        // redo the beta function calculations
        for (int k=0; k<nclusters; k++) {
            double barg1 = alpha(0) + counts_sum[k] + counts_l(i);
            double barg2 = (cluster_counts_(k) + 1.0) * alpha(2) + alpha(1);
            double this_logbeta = lbeta(barg1, barg2);
            log_zprob(k) += logbeta_sum - logbeta[k] + this_logbeta;
        }
    }
}

// add in the contribution to the conditional log-posterior from the markov chain data. this is done in-place.
void ClusterLabels::AddMarkovContribution(arma::vec& log_zprob)
{
    
}

arma::uvec ClusterLabels::RandomPosterior()
{
    arma::uvec zvalues = value_;
    // compute probability of cluster label given others, one-at-a-time
    for (int i=0; i<ndata; i++) {
        arma::vec log_zprob = arma::zeros(nclusters);
        
        // first update the cluster and category counts after removing this data point
        std::vector<int> this_category = RemoveClusterLabel(i);
        
        // start with marginal probability for this cluster label
        AddMarginalContribution(log_zprob);
        
        // now add in contributions from categoricals
        AddCategoricalContribution(log_zprob, this_category);

        // add in contributions from bounded count objects.
        AddBoundedContribution(log_zprob, zvalues, i);
        
        // add in contributions from unbounded count objects. this is basically the same as for the bounded count objects.
        AddUnboundedContribution(log_zprob, zvalues, i);
        
        // TODO: add in contribution from Markov chains for each cluster
        
        // now sample new value of cluster label from categorical distribution
        arma::vec zprob = arma::exp(log_zprob) / arma::sum(arma::exp(log_zprob));
        std::vector<double> stl_zprob = arma::conv_to<std::vector<double> >::from(zprob);
        boost::random::discrete_distribution<> cat_dist(stl_zprob.begin(), stl_zprob.end());
        
        zvalues(i) = cat_dist(rng);
        
        // finally, update the category and cluster counts using this new value for the cluster label
        UpdateClusterCounts(i, zvalues(i));
    }
    
    return zvalues;
}