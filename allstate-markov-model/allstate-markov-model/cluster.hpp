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

class ClusterLabels : public Parameter<arma::uvec> {
    // pointers to parameter objects, either a nparameters x nclusters matrix or a nclusters vector
    std::vector<std::vector<std::shared_ptr<CategoricalPop> > > categoricals_; // list of categorical parameters
    std::vector<std::vector<std::shared_ptr<BoundedCountsPop> > > bounded_counts_; // list of count parameters with upper bound
    std::vector<std::shared_ptr<UnboundedCountsPop> > unbounded_counts_; // count parameter with no upper bound, one per cluster
    std::vector<std::shared_ptr<TransitionMatrix> > transition_matrix_; // markov transition matrix parameter, one per cluster
    
public:
    <#member functions#>
};


#endif /* defined(__allstate_markov_model__cluster__) */
