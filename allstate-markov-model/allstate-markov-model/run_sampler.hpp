//
//  run_sampler.hpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/16/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef allstate_markov_model_run_sampler_hpp
#define allstate_markov_model_run_sampler_hpp

#include <iostream>

// local includes
#include <samplers.hpp>
#include <steps.hpp>
#include "cluster.hpp"
#include "categorical.hpp"
#include "bounded_counts.hpp"
#include "unbounded_counts.hpp"
#include "markov_chain.hpp"


Sampler build_sampler(std::vector<std::vector<int> > categorical_predictors, std::vector<std::vector<int> > bounded_counts,
                      std::vector<std::vector<int> > unbounded_counts, std::vector<std::vector<int> > markov_chain,
                      std::vector<unsigned int> test_set, unsigned int nclusters);

void run_sampler(std::vector<std::vector<int> > categorical_predictors, std::vector<std::vector<int> > bounded_counts,
                 std::vector<std::vector<int> > unbounded_counts, std::vector<std::vector<int> > markov_chain,
                 std::vector<unsigned int> test_set, unsigned int nclusters, unsigned int niter, unsigned int nburn);


#endif
