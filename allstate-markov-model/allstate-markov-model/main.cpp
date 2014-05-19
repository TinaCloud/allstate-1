//
//  main.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/13/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include <iostream>
#include "run_sampler.hpp"
#include "input_output.hpp"
#include <samplers.hpp>
#include <steps.hpp>
#include <proposals.hpp>

int main(int argc, const char * argv[])
{
    int niter = 2;
    int nburnin = 1;
    int nthin = 1;
    
    int nclusters = 10;
    int ntest = 55716;
    int ntrain = 97009;
    int nstates = 2303;
    int ndata = ntest + ntrain;
    // load the data
    std::vector<std::vector<int> > counts;
    int ncount_vars = 6;
    std::string count_fname = "counts_data.dat";
    read_data(count_fname, counts, ndata, ncount_vars);
    
    std::vector<std::vector<int> > categoricals;
    int ncat_vars = 5;
    std::string cat_fname = "cat_data.dat";
    read_data(cat_fname, categoricals, ndata, ncat_vars);
    
    std::vector<std::vector<int> > chains;
    std::string mfname = "markov_chains.dat";
    std::string lfname = "chain_lengths.dat";
    read_markov_data(mfname, lfname, chains, ndata);
    
    std::vector<unsigned int> test_set(ntest);
    for (int i=0; i<ntest; i++) {
        test_set[i] = i + ntrain;
    }
    
    // instantiate the parameter objects
    
    // cluster labels first
    std::shared_ptr<ClusterLabels> Cluster = std::make_shared<ClusterLabels>(true, "Z", nclusters, chains);
    
    // now markov chain parameters
    std::vector<std::shared_ptr<TransitionProbability> > Tprobs;
    for (int k=0; k<nclusters; k++) {
        std::string pname("Tprob-");
        pname += std::to_string(k);
        Tprobs.push_back(std::make_shared<TransitionProbability>(false, pname, chains, nstates, k));
    }
    
    std::vector<std::shared_ptr<TransitionPopulation> > Gammas;
    for (int r=0; r<nstates; r++) {
        for (int c=0; c<nstates; c++) {
            std::string pname("gamma-");
            pname += std::to_string(r);
            pname += "-";
            pname += std::to_string(c);
            Gammas.push_back(std::make_shared<TransitionPopulation>(false, pname, r, c));
        }
    }
    
    std::vector<std::shared_ptr<TransitionHyperPrior> > Hyper;
    for (int i=0; i<nstates + 1; i++) {
        std::string pname("hyper-");
        if (i < nstates) {
            pname += std::to_string(i);
        } else {
            pname += "diag";
        }
        Hyper.push_back(std::make_shared<TransitionHyperPrior>(false, pname));
    }
    
    // bounded counts objects
    std::vector<std::shared_ptr<BoundedCountsPop> > Bcounts;
    for (int l=0; l<counts.size(); l++) {
        std::string pname("bcounts-");
        pname += std::to_string(l);
        unsigned int nmax = 0;
        for (int i=0; i<counts[l].size(); l++) {
            if (counts[l][i] > nmax) {
                nmax = counts[l][i];
            }
        }
        arma::uvec arma_counts;
        arma_counts = arma::conv_to<arma::uvec>::from(counts[l]);
        Bcounts.push_back(std::make_shared<BoundedCountsPop>(false, pname, arma_counts, nmax));
    }
    
    // categorical objects
    std::vector<std::shared_ptr<CategoricalPop> > Cats;
    for (int l=0; l<categoricals.size(); l++) {
        std::string pname("categorical-");
        pname += std::to_string(l);
        arma::uvec arma_cats;
        arma_cats = arma::conv_to<arma::uvec>::from(categoricals[l]);
        Cats.push_back(std::make_shared<CategoricalPop>(false, pname, arma_cats));
    }
    
    // values to predict
    arma::uvec mchain_lengths(chains.size());
    for (int i=0; i<chains.size(); i++) {
        mchain_lengths(i) = chains[i].size();
    }
    Bcounts.push_back(std::make_shared<BoundedCountsPop>(false, "ntime", mchain_lengths, mchain_lengths.max() + 2));
    
    std::vector<std::shared_ptr<MarkovChain> > Mchains;
    for (int l=0; l<test_set.size(); l++) {
        std::string pname("markov_chain-");
        pname += std::to_string(l);
        Mchains.push_back(std::make_shared<MarkovChain>(false, pname, chains, mchain_lengths, test_set[l]));
    }
    
    /*
     *  connect the parameter objects
     */
    
    for (int i=0; i<Mchains.size(); i++) {
        Mchains[i]->SetClusterLabels(Cluster);
        Mchains[i]->SetBoundedCountsPop(Bcounts.back());
        Mchains[i]->SetTransitionMatrix(Tprobs);
    }
    
    // connect the transition matrices for each cluster
    for (int k=0; k<nclusters; k++) {
        Cluster->AddTransitionMatrix(Tprobs[k]);
        Tprobs[k]->SetClusterLabels(Cluster);
        for (int l=0; l<Gammas.size(); l++) {
            Tprobs[k]->AddPopulationPtr(Gammas[l]);
            Gammas[l]->AddTransitionMatrix(Tprobs[k]);
        }
    }
    
    // connect the gammas
    for (int l=0; l<Gammas.size(); l++) {
        int row = Gammas[l]->row_idx;
        for (int m=0; m<Gammas.size(); m++) {
            if (Gammas[m]->row_idx == row && m != l) {
                Gammas[l]->AddGamma(Gammas[m]);
            }
        }
    }
    
    // connect the gammas and their prior
    for (int l=0; l<Gammas.size(); l++) {
        int row = Gammas[l]->row_idx;
        int col = Gammas[l]->col_idx;
        if (row == col) {
            // give gammas along the diagonal their own prior
            Hyper.back()->AddTransitionPop(Gammas[l]);
            Gammas[l]->SetHyperPrior(Hyper.back());
        } else {
            // give all gammas along this column their own prior
            Hyper[col]->AddTransitionPop(Gammas[l]);
            Gammas[l]->SetHyperPrior(Hyper[col]);
        }
    }
    
    // connect the bounded count and categoricals
    for (int l=0; l<Bcounts.size(); l++) {
        Bcounts[l]->SetClusterLabels(Cluster);
        Cluster->AddBoundedCountsPop(Bcounts[l]);
    }
    for (int l=0; l<Cats.size(); l++) {
        Cats[l]->SetClusterLabels(Cluster);
        Cluster->AddCategoricalPop(Cats[l]);
    }
    
    /*
     *  instantiate the Sampler objects and add the steps
     */
    
    Sampler MCMC(niter, nburnin, nthin);
    
    for (int l=0; l<Hyper.size(); l++) {
        MCMC.AddStep(new GibbsStep<arma::vec>(*Hyper[l]));
    }
    
    StudentProposal tProp(8.0, 1.0);
    double ivar = 0.1 * 0.1;
    double target_rate = 0.4;
    for (int l=0; l<Gammas.size(); l++) {
        MCMC.AddStep(new UniAdaptiveMetro(*Gammas[l], tProp, ivar, target_rate, nburnin + niter - 1));
    }
    
    MCMC.AddStep(new GibbsStep<arma::uvec>(*Cluster));
    
    for (int k=0; k<nclusters; k++) {
        MCMC.AddStep(new GibbsStep<arma::mat>(*Tprobs[k]));
    }
    
    for (int l=0; l<Bcounts.size(); l++) {
        MCMC.AddStep(new AdaptiveMetro(*Bcounts[l], tProp, ivar * arma::eye(2, 2), target_rate, nburnin + niter - 1));
    }
    for (int l=0; l<Cats.size(); l++) {
        arma::mat icovar = ivar * arma::eye(Cats[l]->GetNcategories(), Cats[l]->GetNcategories());
        MCMC.AddStep(new AdaptiveMetro(*Cats[l], tProp, icovar, target_rate, nburnin+niter-1));
    }
    
    for (int i=0; i<Mchains.size(); i++) {
        MCMC.AddStep(new GibbsStep<std::vector<int> >(*Mchains[i]));
    }

    MCMC.Run();
}

