__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, betaln

pi = [0.1, 0.25, 0.65]
nx = 1000
probs = [0.25, 0.5, 0.75]
nclusters0 = 3

abeta = bbeta = 0.5
nmax = 100

zcounts0 = np.random.multinomial(nx, pi)

# generate the counts
zvalues = []
counts = []
for k in range(nclusters0):
    zvalues.append(np.zeros(zcounts0[k]) + k)
    counts.append(np.random.binomial(nmax, probs[k], zcounts0[k]))

counts = np.concatenate(counts)
zvalues = np.concatenate(zvalues)

# plt.hist(counts, range=[0, nmax], bins=nmax)
# plt.show()


def zprob(counts, zvalues, idx, abeta, bbeta, nmax, zalpha, nclusters):
    counts_sum0 = np.zeros(nclusters)  # sum of counts per cluster, omitting the index idx
    logbeta0 = np.zeros(nclusters)  # log-beta functions values per cluster before adding in index idx
    zcounts0 = np.zeros(nclusters)
    for k in xrange(nclusters):
        zcounts0[k] = np.sum(zvalues == k)
        counts_sum0[k] = np.sum(counts[zvalues == k])
        if zvalues[idx] == k:
            counts_sum0[k] -= counts[idx]
            zcounts0[k] -= 1
        logbeta0[k] = betaln(counts_sum0[k] + abeta, zcounts0[k] * nmax + bbeta - counts_sum0[k])

    logbeta0_sum = np.sum(logbeta0)
    lnzprob = np.zeros(nclusters)
    for k in xrange(nclusters):
        # calculate lnzprob by adding in data idx for z[idx] = k one at a time so we don't have to redo the beta
        # function calculations
        this_logbeta = betaln(counts_sum0[k] + counts[idx] + abeta,
            (zcounts0[k] + 1) * nmax + bbeta - counts_sum0[k] - counts[idx])
        logbeta_sum = logbeta0_sum - logbeta0[k] + this_logbeta
        lnzprob[k] = np.log(zcounts0[k] + 1 + zalpha / nclusters) + logbeta_sum

    lnzprob -= lnzprob.max()
    zprob = np.exp(lnzprob) / np.sum(np.exp(lnzprob))

    return zprob


def zprob_slow(counts, zvalues, idx, abeta, bbeta, nmax, zalpha, nclusters):
    zvalues_copy = zvalues.copy()
    lnzprob = np.zeros(nclusters)
    for k in range(nclusters):
        zvalues_copy[idx] = k
        lnzprob[k] = np.log(np.sum(zvalues_copy == k) + zalpha / nclusters)
        for k2 in range(nclusters):
            counts_sum = np.sum(counts[zvalues_copy == k2])
            lnzprob[k] += betaln(counts_sum + abeta, np.sum(zvalues_copy == k2) * nmax + bbeta - counts_sum)

    lnzprob -= lnzprob.max()
    zprob = np.exp(lnzprob) / np.sum(np.exp(lnzprob))

    return zprob

# idx = np.random.random_integers(0, nx + 1)
# zp = zprob_slow(counts, zvalues, idx, abeta, bbeta, nmax, 1.0, nclusters0)
# zpf = zprob(counts, zvalues, idx, abeta, bbeta, nmax, 1.0, nclusters0)
# print 'zprobs:', zp
# print zpf
# print 'true z:', zvalues[idx]

# run the MCMC sampler
zvalues0 = zvalues.copy()

nclusters = 3
nsamples = 1000
nburn = 1000
zvalues = np.random.multinomial(1, np.ones(nclusters) / nclusters, nx).argmax(axis=1)
zalpha = 1.0


def run_mcmc_sampler(niter, counts, zvalues, abeta, bbeta, nmax, zalpha, nclusters):
    pi_samples = np.zeros((niter, nclusters))
    psamples = np.zeros((niter, nclusters))
    nx = len(zvalues)
    for i in xrange(niter):
        print i
        for j in xrange(nx):
            zp = zprob(counts, zvalues, j, abeta, bbeta, nmax, zalpha, nclusters)
            zvalues[j] = np.random.multinomial(1, zp).argmax()
        for k in range(nclusters):
            pi_samples[i, k] = np.sum(zvalues == k) / float(len(zvalues))
            psamples[i, k] = np.mean(counts[zvalues == k]) / nmax

    return zvalues, pi_samples, psamples


print 'Burn-in....'
zvalues, pi_samples, psamples = run_mcmc_sampler(nburn, counts, zvalues, abeta, bbeta, nmax, zalpha, nclusters)

print 'Sampling....'
zvalues, pi_samples, psamples = run_mcmc_sampler(nsamples, counts, zvalues, abeta, bbeta, nmax, zalpha, nclusters)

for i in xrange(pi_samples.shape[0]):
    sorted_idx = pi_samples[i].argsort()
    pi_samples[i, :] = pi_samples[i, sorted_idx]
    psamples[i, :] = psamples[i, sorted_idx]

print 'True pi:', pi
print 'Pi-values:'
for k in range(nclusters):
    print np.percentile(pi_samples[:, k], 2.5), np.median(pi_samples[:, k]), np.percentile(pi_samples[:, k], 97.5)

print 'True p:', probs
print 'p-values:'
for k in range(nclusters):
    print np.percentile(psamples[:, k], 2.5), np.median(psamples[:, k]), np.percentile(psamples[:, k], 97.5)

plt.clf()
for k in range(3):
    plt.hist(pi_samples[:, k], bins=50, alpha=0.25, normed=True)
plt.xlabel(r"$\pi$")
plt.show()

plt.clf()
plt.plot(pi_samples)
plt.xlabel('Iteration')
plt.ylabel('PI')
plt.show()

plt.clf()
for k in range(3):
    plt.hist(psamples[:, k], bins=50, alpha=0.25, normed=True)
plt.xlabel("prob")
plt.show()

plt.clf()
plt.plot(psamples)
plt.xlabel('Iteration')
plt.ylabel('probs')
plt.show()

plt.clf()
plt.hist(zvalues, bins=3, range=[0, 2], alpha=0.5)
plt.hist(zvalues0, bins=3, range=[0, 2], alpha=0.5)
plt.xlabel('z')
plt.show()