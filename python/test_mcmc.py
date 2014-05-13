__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

pi = [0.1, 0.25, 0.65]
nx = 1000
nmax = 100
ncats = 4
K = 3

zcounts0 = np.random.multinomial(nx, pi)

alpha0 = 0.5 * np.ones(ncats)
catprobs = np.random.dirichlet(alpha0, K)

# generate the categories
zvalues = []
categories = []
for k in range(K):
    zvalues.append(np.zeros(zcounts0[k]) + k)
    cats = np.random.multinomial(1, catprobs[k, :], size=np.int(zcounts0[k])).argmax(axis=1)
    print cats.shape
    categories.append(cats)

categories = np.concatenate(categories)
zvalues = np.concatenate(zvalues)


def zprob(categories, zvalues, idx, pi, alpha):

    zvalues_copy = zvalues.copy()
    nclusters = len(pi)
    lnzprob = np.zeros(nclusters)
    for k in range(nclusters):
        zvalues_copy[idx] = k
        for k2 in range(nclusters):
            lnzprob[k] += np.log(pi[k2]) + gammaln(np.sum(alpha)) - gammaln(np.sum(zvalues_copy == k2) + np.sum(alpha))
            cats_k = categories[zvalues_copy == k2]
            for j in range(len(alpha)):
                lnzprob[k] += gammaln(np.sum(cats_k == j) + alpha[j]) - gammaln(alpha[j])

    lnzprob -= lnzprob.max()
    zprob = np.exp(lnzprob) / np.sum(np.exp(lnzprob))

    return zprob

# run the MCMC sampler
zvalues0 = zvalues.copy()
np.random.shuffle(zvalues)

