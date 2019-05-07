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
import torch
from tqdm import tqdm
from torch.nn.init import xavier_normal, kaiming_normal
from functools import partial


def relabel(array1, array2):
    if len(array1)==len(array2):
      # set1 is the unique set of array1
        set1 = set(array1)
      # u1 is the unique list of array1
        u1 = list(set1)

      # set2 is the unique set of array2
        set2 = set(array2)
      # set2 is the unique list of array1
        u2 = list(set2)

      #matrix is the Corresponding matrix between u1 and u2
        matrix = [[0 for i in range(len(u2))]for j in range(len(u1))]

        for i in range(len(array1)):
      #item_1 is the index of array1's element in u1
            item_1 = u1.index(array1[i])
          #item_2 is the index of array2's element in u2
            item_2 = u2.index(array2[i])

          #this situation means 1 correspondence between item_1 and item2 is observed
          #so corresponding location in corresponding matrix is incremented
            matrix[item_1][item_2] = matrix[item_1][item_2] + 1

        cost_matrix = benefit_to_cost(matrix)

      #Munkers library solve the cost minimization problem
      #but I would like to solve benefit maximization problem
      #so convert benefit matrix into cost matrix

      #create mukres object
        m = Munkres()
      #get the most corresponded correspondance
        indexes = m.compute(cost_matrix)

      #I use array2 as Integer array so, convert it in case
    array2 = map(int, array2)

      #call replaced function to relace array2 according to object indexes
    replaced_matrix = replaced(array2, u1, u2, indexes)

    return replaced_matrix

def transpose(array):
    return list(map(list, zip(*array)))

def relabel_cluster(clusters):
    #use first object in list object clusters as criteria
    criteria = clusters[0]

    # M is the number of review in each clustering
    M = len(criteria)
    
    # N is the number of clustering
    N = len(clusters)
    
    for idx in range(1,N):
        #if wrong size of clustering appears, stop the process
        if len(clusters[idx]) != M:
            print ("Cluster "+str(idx)+" is out of size")
            return -1
        clusters[idx] = relabel(criteria, clusters[idx])
    return clusters

def benefit_to_cost(matrix):
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row = cost_row + [(sys.maxsize - col)]
        cost_matrix = cost_matrix + [cost_row]
    return cost_matrix

def replaced(array, u1, u2, cor):
    #copy array for isolation
    #Need to use deepcopy because of the nature of Python for treating 
    array=list(array)
    replaced = copy.deepcopy(array)
    #cor is the corresponding list
    for row, col in cor:
        #u1[row] and u2[col] is corresponded
        for idx in range(len(array)):
            #if the element of array is equal to u2[col]
            if array[idx] == u2[col]:
                #the element is corresponding to u1[row]
                #so isolated list replaced is replaced by u1[row]
                replaced[idx] = u1[row]
    return replaced
                

def voting(clusters):
    #Transpose Clusters
    print(np.shape(clusters))
    clusters = transpose(clusters)
    print(np.shape(clusters))
    print(len(clusters))
    print(len(np.unique(clusters)))
    counter= np.zeros(shape=(len(clusters),len(np.unique(clusters))))
    b=0
    for i in clusters:
        print('b=',b)
        a=0
        for j in np.unique(clusters):
            print('a=',a)
            counter[b][a]=i.count(j)
            a=a+1
        b=b+1
    return counter






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

    data = torch.Tensor(list_ensembles)
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
     

def do_voting_gmm_torch(list_ensembles, nEnsCluster, iterations, verbose, N_clusters_max, hdf5_file_name, data):
    relabeled_clusters=relabel_cluster(list_ensembles)
    vot_matrix=voting(relabeled_clusters)
    mixtureObj = emProcess(vot_matrix, nEnsCluster, iterations)
    return mixtureObj