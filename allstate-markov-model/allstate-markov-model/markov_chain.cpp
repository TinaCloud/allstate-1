//
//  markov_chain.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/15/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include <boost/math/special_functions/gamma.hpp>

#include "markov_chain.hpp"
#include "cluster.hpp"

using boost::math::lgamma;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

/*
 *  FUNCTION DEFINITIONS FOR TRANSITION MATRIX
 */


TransitionProbability::TransitionProbability(bool track, std::string label, std::vector<std::vector<int> >& data,
                                             unsigned int ncats, unsigned int k, double temperature) :
                            Parameter<arma::mat>(track, label, temperature), ncategories(ncats), data_(data), cluster_id(k)
{
    ndata = data_.size();
    // get the number of time points for each datum
    for (int i=0; i<ndata; i++) {
        ntime[i] = data_[i].size();
    }
}

arma::mat TransitionProbability::StartingValue()
{
    return RandomPosterior();
}

// return a draw from the conditional posterior, p(T|data, gamma, Z)
arma::mat TransitionProbability::RandomPosterior()
{
    arma::uvec zvalues = cluster_labels_->Value();
    
    // build matrix of transition counts
    arma::mat transition_counts = arma::zeros<arma::mat>(ncategories, ncategories);
    
    // TODO: this may be expensive: might need to update the counts when updating the cluster labels.
    
    // first do data contribution
    for (int i=0; i<ndata; i++) {
        if (zvalues(i) == cluster_id) {
            // only count transitions for data in this cluster
            for (int t=1; t<ntime[i]; t++) {
                int row_id = data_[i][t-1];
                int col_id = data_[i][t];
                // transition from category row_id to category col_id over t-1 -> t.
                transition_counts(row_id, col_id)++;
            }
        }
    }
    
    // now add population-level contribution
    for (int j=0; j<ncategories * ncategories; j++) {
        double gamma = exp(population_par_[j]->Value());  // gamma values are sampled on the log scale since they must be positive
        unsigned int row_id = population_par_[j]->row_idx;
        unsigned int col_id = population_par_[j]->col_idx;
        transition_counts(row_id, col_id) += gamma;
    }

    arma::mat Tprop(ncategories, ncategories);
    // obtain draws from a dirichlet distribution, one row at a time
    for (int r=0; r<ncategories; r++) {
        arma::vec gdraws(ncategories);
        for (int c=0; c<ncategories; c++) {
            gdraws[c] = RandGen.gamma(transition_counts(r,c), 1.0);
        }
        double gdraws_sum = arma::sum(gdraws);
        Tprop.row(r) = gdraws / gdraws_sum;
    }

    return Tprop;
}


/*
 *  FUNCTION DEFINITIONS FOR POPULATION-LEVEL PARAMETERS OF THE TRANSITION MATRICES (AKA, THE GAMMAS)
 */

TransitionPopulation::TransitionPopulation(bool track, std::string label, unsigned int r, unsigned int c, double temperature) :
    Parameter<double>(track, label, temperature), row_idx(r), col_idx(c) {}

// initialize by drawing from the prior
double TransitionPopulation::StartingValue()
{
    arma::vec prior_values = hyper_prior_->Value();
    double prior_mean = prior_values(0);
    double prior_var = prior_values(1);
    double log_gamma = RandGen.normal(prior_mean, sqrt(prior_var));
    return log_gamma;
}

// return the conditional log-posterior: log p(gamma_ij|T_1, ..., T_K, prior values, gamma_ik for k \neq j).
double TransitionPopulation::LogDensity(double log_gamma)
{
    double gamma = exp(log_gamma);
    double gamma_sum = gamma; // sum over all gammas for this row, needed for the normalization of the dirichlet distribution
    for (int j=0; j<gammas_this_row.size(); j++) {
        gamma_sum += exp(gammas_this_row[j]->Value());
    }
    double logdensity;
    
    // first add contribution from hyper-prior
    double prior_mean = hyper_prior_->Value()(0);
    double prior_var = hyper_prior_->Value()(1);
    logdensity = -0.5 * (log_gamma - prior_mean) * (log_gamma - prior_mean) / prior_var;
    
    // now add contribution from dirichlet likelihood
    unsigned int nclusters = transition_matrices_.size();
    logdensity += nclusters * (lgamma(gamma_sum) - lgamma(gamma));
    double log_tprob_sum = 0.0;
    for (int k=0; k<nclusters; k++) {
        log_tprob_sum += log(transition_matrices_[k]->Value()(row_idx, col_idx));
    }
    logdensity += (gamma - 1.0) * log_tprob_sum;
    
    return logdensity;
}




