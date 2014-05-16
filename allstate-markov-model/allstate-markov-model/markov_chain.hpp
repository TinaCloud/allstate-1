//
//  markov_chain.h
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/15/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef __allstate_markov_model__markov_chain__
#define __allstate_markov_model__markov_chain__

#include <iostream>
#include <vector>
#include <memory>

// local includes
#include <parameters.hpp>
#include <armadillo>

// foward declarations
class ClusterLabels;
class TransitionPopulation;
class TransitionHyperPrior;

/*
 *  CLASS FOR THE CLUSTER-SPECIFIC TRANSITION PROBABILITY OBJECT
 */

class TransitionProbability : public Parameter<arma::mat> {
    std::vector<std::vector<int> >& data_;  // the set of time series, can have different lengths
    std::shared_ptr<ClusterLabels> cluster_labels_;  // pointer to the cluster labels object
    // pointer to the population-level parameter objects, in matrix format
    std::vector<std::shared_ptr<TransitionPopulation> > population_par_;
public:
    unsigned int ndata;  // the number of data points
    std::vector<unsigned int> ntime;  // the number of time points for each data point
    unsigned int ncategories;  // the number of possible category values. assumes category labels are 0, 1, ..., ncategories-1
    unsigned int cluster_id;  // the label of the cluster that this parameter corresponds to
    
    TransitionProbability(bool track, std::string label, std::vector<std::vector<int> >& data, unsigned int ncats,
                          unsigned int k, double temperature=1.0);
    
    arma::mat StartingValue();
    
    arma::mat RandomPosterior();
    
    void SetClusterLabels(std::shared_ptr<ClusterLabels> labels) { cluster_labels_ = labels; }
    
    void SetPopulationPtr(std::shared_ptr<TransitionPopulation> gamma) {
        population_par_.push_back(gamma);
    }
};

/*
 *  CLASS FOR EACH ELEMENT OF THE POPULATION-LEVEL PARAMETER OF THE TRANSITION MATRICES
 */

class TransitionPopulation : public Parameter<double> {
    std::vector<std::shared_ptr<TransitionProbability> > transition_matrices_;  // pointer to the transition probability objects for each cluster
    std::vector<std::shared_ptr<TransitionPopulation> > gammas_this_row;  // pointer to other populationa-level parameters in this row
    std::shared_ptr<TransitionHyperPrior> hyper_prior_;  // pointer to the hyper-prior parameters corresponding to this object
public:
    unsigned int row_idx;  // make sure this object knows which row and column of the transition matrix it corresponds to
    unsigned int col_idx;
    
    TransitionPopulation(bool track, std::string label, unsigned int r, unsigned int c, double temperature=1.0);
    
    double StartingValue();
    
    double LogDensity(double gamma);
    
    void AddTransitionMatrix(std::shared_ptr<TransitionProbability> tmatrix) {
        transition_matrices_.push_back(tmatrix);
    }
    
    void AddGamma(std::shared_ptr<TransitionPopulation> other_gamma) {
        gammas_this_row.push_back(other_gamma);
    }
    
    void SetHyperPrior(std::shared_ptr<TransitionHyperPrior> hyper_prior) { hyper_prior_ = hyper_prior; }
};

/*
 *  CLASS FOR HYPER-PRIOR OF THE POPULATION-LEVEL PARAMETERS OF THE TRANSITION MATRICES, A LOG-NORMAL DISTRIBUTION FOR EACH COLUMN AND ALONG THE
 *  DIAGONAL.
 */

class TransitionHyperPrior : public Parameter<arma::vec> {
    // pointer to the set of population-level elements of the transition matrix governed by this hyperprior
    std::vector<std::shared_ptr<TransitionPopulation> > gammas;
public:
    double prior_mean;  // prior mean for the mean on log(gamma)
    double prior_ndata;  // reduction factor for the prior variance on the mean on log(gamma)
    unsigned int prior_dof;  // prior degrees-of-freedom on the variance of log(gamma)
    double prior_ssqr;  // prior scale parameter on the variance of log(gamma)
    
    TransitionHyperPrior(bool track, std::string label, double prior_mu, double prior_kappa, unsigned int prior_nu,
                         double prior_var0, double temperature=1.0);
    
    arma::vec StartingValue();
    
    arma::vec RandomPosterior();
    
    void AddTransitionPop(std::shared_ptr<TransitionPopulation> gamma) { gammas.push_back(gamma); }
};


#endif /* defined(__allstate_markov_model__markov_chain__) */
