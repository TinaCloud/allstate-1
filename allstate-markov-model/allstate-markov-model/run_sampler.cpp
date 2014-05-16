//
//  run_sampler.c
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/15/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

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
                      std::vector<std::vector<int> > unbounded_counts, unsigned int nclusters)
{
    
}

void run_sampler(std::vector<std::vector<int> > categorical_predictors, std::vector<std::vector<int> > bounded_counts,
                 std::vector<std::vector<int> > unbounded_counts, unsigned int nclusters, unsigned int niter, unsigned int nburn)
{
    
}