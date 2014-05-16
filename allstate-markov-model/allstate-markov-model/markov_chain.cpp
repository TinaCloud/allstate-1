//
//  markov_chain.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/15/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include "markov_chain.hpp"
#include "cluster.hpp"


// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;


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
        double gamma = population_par_[j]->Value();
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













