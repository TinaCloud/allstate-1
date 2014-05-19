//
//  missing_data.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/18/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include "missing_data.hpp"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;


MarkovChain::MarkovChain(bool track, std::string label, std::vector<std::vector<int> >& data, arma::uvec& ntime,
                         unsigned int i, double temperature) : Parameter<std::vector<int> >(track, label, temperature),
                        idx(i), data_(data), ntime_(ntime)
{
    nobserved = ntime_(idx);  // number of observed data points in the markov chain
}

std::vector<int> MarkovChain::StartingValue() {
    int nt = boost::random::uniform_int_distribution<>(nobserved+1, bounded_counts_->nmax)(rng);
    unsigned int this_cluster = cluster_labels_->Value()(idx);

    // now simulate values of the markov chain given this length
    arma::mat Tprob = Tmat_[this_cluster]->Value();
    int nstates = Tprob.n_cols;
    
    std::vector<int> mchain = data_[idx];
    mchain.resize(nt);
    
    for (int t=nobserved; t<nt; t++) {
        int previous_state = data_[idx][t-1];
        std::vector<double> tprob = arma::conv_to<std::vector<double> >::from(Tmat_[this_cluster]->Value().row(previous_state));
        boost::random::discrete_distribution<> markov(tprob.begin(), tprob.end());
        mchain[t] = markov(rng);
    }
    
    return mchain;

}

std::vector<int> MarkovChain::RandomPosterior() {
    // first calculate the discrete probabilities of the length of the markov chain
    unsigned int nmax = bounded_counts_->nmax;
    int ngrid = nmax - nobserved;

    double counts_sum = 0.0;
    unsigned int this_cluster = cluster_labels_->Value()(idx);
    double zcounts_k = cluster_labels_->GetClusterCounts()(this_cluster);
    for (int i=0; i<ntime_.n_elem; i++) {
        if (cluster_labels_->Value()(i) == this_cluster) {
            counts_sum++;
        }
    }
    
    arma::vec log_probs(ngrid);
    arma::vec alpha = arma::exp(bounded_counts_->Value());
    for (int j=0; j<ngrid; j++) {
        double new_counts = nobserved + j + 1;
        double new_counts_sum = counts_sum + new_counts;
        log_probs[j] = -lbeta(nmax - new_counts + 1.0, new_counts + 1.0) +
            lbeta(new_counts_sum + alpha(0), alpha(1) + nmax * zcounts_k - new_counts_sum);
    }
    arma::vec probs = arma::exp(log_probs - log_probs.max()) / arma::sum(arma::exp(log_probs - log_probs.max()));
    
    std::vector<double> stl_probs = arma::conv_to<std::vector<double> >::from(probs);
    boost::random::discrete_distribution<> nt_dist(stl_probs.begin(), stl_probs.end());
    
    // simulate new value of the length of the markov chain, conditional on it being greater than nobserved
    unsigned int new_ntime = nobserved + 1 + nt_dist(rng);
    
    // now simulate new values of the markov chain given this length
    arma::mat Tprob = Tmat_[this_cluster]->Value();
    int nstates = Tprob.n_cols;
    
    std::vector<int> mchain = data_[idx];
    mchain.resize(new_ntime);
    
    for (int t=nobserved; t<new_ntime; t++) {
        int previous_state = mchain[t-1];
        std::vector<double> tprob = arma::conv_to<std::vector<double> >::from(Tmat_[this_cluster]->Value().row(previous_state));
        boost::random::discrete_distribution<> markov(tprob.begin(), tprob.end());
        mchain[t] = markov(rng);
    }

    return mchain;
}