__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from numba import autojit

pi = [0.1, 0.25, 0.65]
nx = 1000
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
    categories.append(cats)

categories = np.concatenate(categories)
zvalues = np.concatenate(zvalues)


def update_zvalues(categories, zvalues, alpha, zalpha, nclusters):
    nx = len(zvalues)
    ncategories = 1 + categories.max()  # assumes categories have values from j=0, ..., J-1
    n_k = np.zeros(nclusters)
    n_jk = np.zeros((ncategories, nclusters))
    # get initial counts
    for k in xrange(nclusters):
        n_k[k] = np.sum(zvalues == k)
        for j in xrange(ncategories):
            n_jk[j, k] = np.sum(categories[zvalues == k] == j)
    alpha_sum = np.sum(alpha)

    # now update the values
    for i in xrange(nx):
        # remove contribution to counts from this source
        this_cluster = zvalues[i]
        this_category = categories[i]
        n_k[this_cluster] -= 1
        n_jk[this_category, this_cluster] -= 1

        logzp = np.log(n_k + zalpha / nclusters) + np.log(n_jk[this_category] + alpha[this_category]) - \
            np.log(n_k + alpha_sum)

        logzp -= np.min(logzp)
        zp = np.exp(logzp) / np.sum(np.exp(logzp))

        # zprob0 = zprob(categories, zvalues, i, alpha, zalpha, nclusters)
        # assert np.allclose(zp, zprob0)

        # draw new cluster label
        zvalues[i] = np.random.multinomial(1, zp).argmax()

        # now update the counts
        n_k[zvalues[i]] += 1
        n_jk[this_category, zvalues[i]] += 1

    return zvalues


def zprob(categories, zvalues, idx, alpha, zalpha, nclusters):
    lnzprob = np.zeros(nclusters)
    for k in range(nclusters):
        n_jk = np.sum(categories[zvalues == k] == categories[idx])
        n_k = np.sum(zvalues == k)
        if zvalues[idx] == k:
            n_jk -= 1  # make sure we do not count the data point we are updating
            n_k -= 1
        lnzprob[k] = np.log(n_k + zalpha / nclusters) + np.log(n_jk + alpha[categories[idx]]) - \
            np.log(n_k + np.sum(alpha))

    lnzprob -= np.max(lnzprob)
    zprob = np.exp(lnzprob) / np.sum(np.exp(lnzprob))

    return zprob


def zprob_slow(categories, zvalues, idx, alpha, zalpha, nclusters):

    zvalues_copy = zvalues.copy()
    lnzprob = np.zeros(nclusters)
    for k in range(nclusters):
        zvalues_copy[idx] = k
        for k2 in range(nclusters):
            lnzprob[k] += gammaln(np.sum(alpha)) - gammaln(np.sum(zvalues_copy == k2) + np.sum(alpha))
            lnzprob[k] += gammaln(np.sum(zvalues_copy == k2) + zalpha / nclusters)
            cats_k = categories[zvalues_copy == k2]
            for j in range(len(alpha)):
                lnzprob[k] += gammaln(np.sum(cats_k == j) + alpha[j]) - gammaln(alpha[j])

    lnzprob -= lnzprob.max()
    zprob = np.exp(lnzprob) / np.sum(np.exp(lnzprob))

    return zprob


# find most probable cluster
# nclusters = 3
# zhat = np.zeros(nx)
# pi_hat = np.zeros(nclusters)
# for i in range(nx):
#     zp = zprob(categories, zvalues, i, alpha0, 1.0, nclusters)
#     zhat[i] = zp.argmax()
#     pi_hat += zp
#
# pi_hat /= nx
#
# plt.plot(pi, '-o')
# plt.plot(pi_hat, '-o')
# plt.show()

# run the MCMC sampler
zvalues0 = zvalues.copy()

nclusters = 3
nsamples = 1000
nburn = 1000
zvalues = np.random.multinomial(1, np.ones(nclusters) / nclusters, nx).argmax(axis=1)
zalpha = 1.0


def run_mcmc_sampler(niter, zvalues, alpha, zalpha, nclusters):
    pi_samples = np.zeros((nclusters, niter))
    for i in xrange(niter):
        print i
        zvalues = update_zvalues(categories, zvalues, alpha, zalpha, nclusters)
        for k in range(nclusters):
            pi_samples[k, i] = np.sum(zvalues == k) / float(len(zvalues))

    return zvalues, pi_samples


print 'Burn-in....'

zvalues, pi_samples = run_mcmc_sampler(nburn, zvalues, alpha0, zalpha, nclusters)

print 'Sampling....'

zvalues, pi_samples = run_mcmc_sampler(nsamples, zvalues, alpha0, zalpha, nclusters)

pi_samples.sort(axis=0)

print 'True pi:', pi
print 'Pi-values:'
for k in range(nclusters):
    print np.percentile(pi_samples[k, :], 2.5), np.median(pi_samples[k, :]), np.percentile(pi_samples[k, :], 97.5)


plt.clf()
for k in range(3):
    plt.hist(pi_samples[k, :], bins=50, alpha=0.25, normed=True)
plt.xlabel(r"$\pi$")
plt.show()

plt.clf()
plt.plot(pi_samples.T)
plt.xlabel('Iteration')
plt.ylabel('PI')
plt.show()

plt.clf()
plt.hist(zvalues, bins=3, range=[0, 2], alpha=0.5)
plt.hist(zvalues0, bins=3, range=[0, 2], alpha=0.5)
plt.xlabel('z')
plt.show()