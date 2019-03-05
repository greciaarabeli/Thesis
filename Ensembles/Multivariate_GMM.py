import numpy as np
import pandas as pd
import sys
import random
import scipy as sc
import random as rand
from sys import maxsize


def PosSymDefMatrix(n,sd):
    M = np.matrix(np.random.rand(n,n))
    M = 0.5*(M + M.T)
    M = M + sd*np.eye(n)
    return M

def EStep(X, w, mu, cov, N, M, H):
    
    r_ij = np.zeros((N, M))

    for i in range(N):
        
        r_ij_Sumj = np.zeros(M)
        
        for j in range(M):
            
            r_ij_Sumj[j] = w[j] * sc.stats.multivariate_normal.pdf(X[i,:], mu[j,:], cov[:,:,j])
            
        for j in range(M):
            r_ij[i,j] =   r_ij_Sumj[j] / np.sum(r_ij_Sumj)
    
    return r_ij


def MStep(r, X, mu, cov, N,M,H):
    
    # the weigths
    w_j = np.sum(r, axis=0) / N
    
    Allmu_j = np.zeros((N, M, H))
    Allcov_j = np.zeros((N, H, H, M))
    
    # mean
    for i in range(N):
        
        Allmu_j[i,:,:] = np.outer(r[i,:], X[i,:])
    
    mu_j = np.zeros((M, H))
    
    for j in range(M):
        mu_j[j,:] = (1/np.sum(r, axis=0)[j]) * np.sum(Allmu_j, axis=0)[j,:]
        
    # sd
    for i in range(N):
        for j in range(cov.shape[2]):
            Allcov_j[i,:,:,j] = r[i,j] * np.outer((X[i,:] - mu_j[j,:]), (X[i,:]-mu_j[j,:]))

    cov_j = np.zeros((cov.shape[0], cov.shape[1], cov.shape[2]))
    
    for j in range(cov.shape[2]):
        
        cov_j[:,:,j] = (1/np.sum(r, axis=0)[j]) * np.sum(Allcov_j, axis=0)[:,:,j]
    
    return w_j,mu_j,cov_j



def emProcess(list_ensembles, nEnsCluster, iterations):
    N=list_ensembles.shape[1] 
    H = len(list_ensembles)
    M=nEnsCluster

    listParts = np.concatenate(list_ensembles).reshape(H,N)
    df = [] 
    [df.append(listParts[:,i]) for i in range(N)]

    df = pd.DataFrame(df, columns= np.arange(H))
    X=df.values
    
    # Initialize parameters randomly
    sdDiff = 2
    SDClass = np.random.rand(1,M)+sdDiff
    Cov = [PosSymDefMatrix(H,i) for i in SDClass[0]]
    broadness = 2

    initMu = np.empty([M, H])
    initCov = np.empty([H, H, M])

    for j in range(M):

        initMu[j,:] = np.random.random(H)*np.amax(X, axis=0)
        initCov[:,:,j] = np.mean(np.array(Cov), axis=0)+broadness

    initw=np.ones(M) / M
    
    
    Initializations = iterations
    EMiteration = iterations
    lookLH = 20

    for init in range(Initializations):
         # starting values
        initMu = np.empty([M, H])
        for j in range(M):
            initMu[j,:] = np.random.random(H)*np.amax(X, axis=0)

        r_n = EStep(X, initw, initMu, initCov, N,M, H)
        w_n,mu_n,cov_n = MStep(r_n, X, initMu, initCov, N, M, H)

        if init == 0:
            logLH = -1000000000000

        for i in range(EMiteration):

            # E step
            r_n = EStep(X, w_n, mu_n, cov_n, N,M,H)

            # M step
            w_n,mu_n,sigma_n = MStep(r_n, X, mu_n, cov_n, N,M,H)

            # compute log likelihood
            logLall = np.zeros((N))

            for i in range(N):

                LH = np.zeros(M)

                for jClass in range(M):
                    LH[jClass] = w_n[jClass] * sc.stats.multivariate_normal.pdf(X[i,:], mu_n[jClass,:], cov_n[:,:,jClass])

                logLall[i] = np.log(np.sum(LH))

            logL = np.sum(logLall)

            if i > EMiteration - lookLH:
                print (logL)

        if logL > logLH:
            logLH = logL
            print ('found larger: ', logLH)
            w_p = w_n
            mu_p = mu_n
            sigma_p = sigma_n
            r_p = r_n
            
    final_prob=EStep(X, w_p, mu_p, sigma_p, N, M, H)
    
    final_part=np.empty([N])
    for i in range(N):
        print(i)
        final_part[i]=np.argmax(final_prob[i])
        
    return final_part

'''Input    
    list ensemble: array of arrays wit the labels of each clustering
    nEnsCluster: number of clusters to find
    iterations: max number of iterations for EM Process '''

def do_gmm(list_ensembles, nEnsCluster, iterations, verbose, N_clusters_max, hdf5_file_name):
    mixtureObj = emProcess(list_ensembles, nEnsCluster, iterations)
    return mixtureObj
