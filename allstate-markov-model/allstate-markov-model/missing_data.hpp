//
//  missing_data.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/18/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__missing_data__
#define __allstate_markov_model__missing_data__

#include <iostream>
#include "cluster.hpp"
#include "bounded_counts.hpp"
#include "markov_chain.hpp"


class MarkovChain : public Parameter<std::vector<int> > {
    std::vector<std::vector<int> >& data_;
    arma::uvec& ntime_;
    std::shared_ptr<ClusterLabels> cluster_labels_;
    std::vector<std::shared_ptr<TransitionProbability> > Tmat_;
    std::shared_ptr<BoundedCountsPop> bounded_counts_;
    
public:
    unsigned int ntime;
    unsigned int nobserved;
    unsigned int idx;
    
    MarkovChain(bool track, std::string label, std::vector<std::vector<int> >& data, arma::uvec& ntime,
                unsigned int idx, double temperature=1.0);
    
    std::vector<int> StartingValue();
    
    std::vector<int> RandomPosterior();
    
    void Save(std::vector<int> new_value) {
        value_ = new_value;
        data_[idx] = new_value;  // update the data for this markov chain
        ntime_[idx] = new_value.size();
    }
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels> labels) { cluster_labels_ = labels; }
    void SetBoundedCountsPop(std::shared_ptr<BoundedCountsPop> bounded_counts) {
        assert(nobserved < bounded_counts->nmax);
        bounded_counts_ = bounded_counts;
    }
    void SetTransitionMatrix(std::vector<std::shared_ptr<TransitionProbability> > Tmat) { Tmat_ = Tmat; }
    
};

#endif /* defined(__allstate_markov_model__missing_data__) */
