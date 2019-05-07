import pandas as pd
import sys
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import random as rand
from sys import maxsize

import numpy as np
import pandas as pd
from munkres import Munkres #Hungarian algorithm
import sys
import copy
import bisect
from itertools import combinations
import torch
from tqdm import tqdm
from torch.nn.init import xavier_normal, kaiming_normal
from functools import partial

def do_consensus_matrix(c_list):
    Mk = np.zeros((len(c_list[0]), len(c_list[0])))
    Is = np.zeros((len(c_list[0]),)*2)


    # find indexes of elements from same clusters with bisection
    # on sorted array => this is more efficient than brute force search
    a=0
    for c in c_list:
        id_clusts = np.argsort(c)
        sorted_ = c[id_clusts]
        for i in np.unique(c): 
            ia = bisect.bisect_left(sorted_, i)
            ib = bisect.bisect_right(sorted_, i)
            is_ = id_clusts[ia:ib]
            ids_ = np.array(list(combinations(is_, 2))).T
                # sometimes only one element is in a cluster (no combinations)
            if ids_.size != 0:
                Mk[ids_[0], ids_[1]] += 1

    # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
    Mk+= Mk.T
    Mk[range(len(c_list[0])), range(len(c_list[0]))] = 1

    return Mk


def initialize(data, k, d, var=1):
    """
    :param data: design matrix (examples, features)
    :param K: number of gaussians
    :param var: initial variance
    """
  # choose k points from data to initialize means
    m = data.size(0)
    idxs = torch.from_numpy(np.random.choice(m, k, replace=False))
    mu = data[idxs]

  # uniform sampling for means and variances
    var = torch.Tensor(k, d).fill_(var)

  # uniform prior
    pi = torch.empty(k).fill_(1. / k)

    return mu, var, pi




def log_gaussian(x, mean, logvar,log_norm_constant):
    """
    Returns the density of x under the supplied gaussian. Defaults to
    standard gaussian N(0, I)
    :param x: (*) torch.Tensor
    :param mean: float or torch.FloatTensor with dimensions (*)
    :param logvar: float or torch.FloatTensor with dimensions (*)
    :return: (*) elementwise log density
    """
    if type(logvar) == 'float':
        logvar = x.new(1).fill_(logvar)

    a = (x - mean) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = log_p + log_norm_constant

    return log_p


def get_likelihoods(X, mu, logvar, log_norm_constant, log=True):
    """
    :param X: design matrix (examples, features)
    :param mu: the component means (K, features)
    :param logvar: the component log-variances (K, features)
    :param log: return value in log domain?
      Note: exponentiating can be unstable in high dimensions.
    :return likelihoods: (K, examples)
    """

    # get feature-wise log-likelihoods (K, examples, features)
    log_likelihoods = log_gaussian(
    X[None, :, :], # (1, examples, features)
    mu[:, None, :], # (K, 1, features)
    logvar[:, None, :], # (K, 1, features)
    log_norm_constant)

  # sum over the feature dimension
    log_likelihoods = log_likelihoods.sum(-1)

    if not log:
        log_likelihoods.exp_()

    return log_likelihoods





def logsumexp(x, dim, keepdim=False):
    """
    :param x:
    :param dim:
    :param keepdim:
    :return:
    """
    max, _ = torch.max(x, dim=dim, keepdim=True)
    out = max + (x - max).exp().sum(dim=dim, keepdim=keepdim).log()
    return out


def get_posteriors(log_likelihoods, log_pi):
    """
    Calculate the the posterior probabities log p(z|x), assuming a uniform prior over
    components (for this step only).
    :param likelihoods: the relative likelihood p(x|z), of each data point under each mode (K, examples)
    :param log_pi: log prior (K)
    :return: the log posterior p(z|x) (K, examples)
    """
    posteriors = log_likelihoods # + log_pi[:, None]
    posteriors = posteriors - logsumexp(posteriors, dim=0, keepdim=True)
    return posteriors


def get_parameters(X, log_posteriors, eps=1e-6, min_var=1e-6):
    """
    :param X: design matrix (examples, features)
    :param log_posteriors: the log posterior probabilities p(z|x) (K, examples)
    :returns mu, var, pi: (K, features) , (K, features) , (K)
    """
    posteriors = log_posteriors.exp()

  # compute `N_k` the proxy "number of points" assigned to each distribution.
    K = posteriors.size(0)
    N_k = torch.sum(posteriors, dim=1) # (K)
    N_k = N_k.view(K, 1, 1)

  # get the means by taking the weighted combination of points
  # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
    mu = posteriors[:, None] @ X[None,]
    mu = mu / (N_k + eps)

  # compute the diagonal covar. matrix, by taking a weighted combination of
  # the each point's square distance from the mean
    A = X - mu
    var = posteriors[:, None] @ (A ** 2) # (K, 1, features)
    var = var / (N_k + eps)
    logvar = torch.clamp(var, min=min_var).log()

  # recompute the mixing probabilities
    m = X.size(1) # nb. of training examples
    pi = N_k / N_k.sum()

    return mu.squeeze(1), logvar.squeeze(1), pi.squeeze()

def emProcess(list_ensembles, nEnsCluster, iterations):
    
    # training loop

    data = torch.Tensor(list_ensembles.T)
    N=len(data)
    d=len(data[0])
    mu, var, pi = initialize(data, nEnsCluster, d, var=1)
    logvar = var.log()

    prev_cost = float('inf')
    thresh = 1e-5
    for i in tqdm(range(iterations)):
        # get the likelihoods p(x|z) under the parameters

        log_norm_constant = -0.5 * np.log(2 * np.pi)

        log_likelihoods = get_likelihoods(data, mu, logvar, log_norm_constant)

         # compute the "responsibilities" p(z|x)
        log_posteriors = get_posteriors(log_likelihoods, pi.log())

        # compute the cost and check for convergence
        cost = log_likelihoods.mean()
        diff = prev_cost - cost
        if torch.abs(diff).item() < thresh:
            break
        prev_cost = cost

        # re-compute parameters
        mu, logvar, pi = get_parameters(data, log_posteriors)
        
    posteriors =log_posteriors.exp_()
    posteriors_np=posteriors.numpy()
    
    final_part=np.empty([N])
    for i in range(N):
        final_part[i]=np.argmax(posteriors_np.T[i])
        
    return final_part

def do_consensus_gmm_torch(list_ensembles, nEnsCluster, iterations, verbose, N_clusters_max, hdf5_file_name, data):
    consensus=do_consensus_matrix(list_ensembles) 
    mixtureObj = emProcess(consensus, nEnsCluster, iterations)
    return mixtureObj
