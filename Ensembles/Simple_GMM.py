import numpy as np
import pandas as pd
import sys
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random as rand
from sys import maxsize





def PosSymDefMatrix(n,sd):
    M = np.matrix(np.random.rand(n,n))
    M = 0.5*(M + M.T)
    M = M + sd*np.eye(n)
    return M





def distance(old_params, new_params, N, M, H):
    dist = 0
    for i in range(M):
        for j in range(H):
            dist+=(old_params['mu'][0][i][j]-new_params['mu'][0][i][j])**2
    return dist**0.5


def prob(val, mu, sig, lam):
    p = lam
    for i in range(len(val)):
        p *= norm.pdf(val[i], mu[i], sig[i][i])
    return p

def expectation(dataFrame, parameters,N, M, H):
    p_cluster=pd.DataFrame(0,range(N),range(M))
    for i in range(N):
        print('i=',i)
        for j in range(M):
            print('j=',j)
            p_cluster.loc[[i],[j]]=prob(np.array(dataFrame.iloc[i,:H]), list(parameters['mu'][0][j]), list(parameters['sig'][0][j]), parameters['lambda'][0][j])
        dataFrame['label'][i]=np.argmax(p_cluster.iloc[[i]].values)
    return dataFrame

def maximization(dataFrame, parameters, N, M, H):
    len_points=np.zeros(M)
    for i in range(M):
        points=dataFrame[dataFrame['label']==i]
        len_points[i]=len(points)
        print(len_points)
        mu=np.zeros(H)
        sig=np.zeros([H, H])
        for j in range (H):
            mu[j]=points[j].mean()
            for k in range (H):
                if j==k:
                    sig[j][k]=points[j].std()
                else:
                    sig[j][k]=0

        parameters['mu'][0][i]=mu
        parameters['sig'][0][i]=sig
    parameters['lambda'][0]=len_points/N
    return parameters

def concatenate(list_ensembles):
    l=list_ensembles.T
    voting_matrix=np.empty([l.shape[0]])
    for i in range(l.shape[0]):
        print(i)
        a=str(int(l[i][0]))
        for j in range(1,l.shape[1]):
            print(j)
            a=a+str(int(l[i][j]))
        voting_matrix[i]=int(a)
    return voting_matrix


def emProcess(list_ensembles,nEnsCluster, iterations):
    #Choose three random index no's from the length of dataset
    
    voting_matrix=concatenate(list_ensembles)
    N=len(voting_matrix) 
    H = 1
    M=nEnsCluster

    df = pd.DataFrame(voting_matrix, columns= np.arange(H))
    df.index.name = 'objs'
    df.columns.name = 'partition'
    X=df.values
    #Initialize parameters

    k=[random.randrange(len(X)) for _ in range(M)]
    mu=[M,H]
    lam=np.ones(M) / M

    Cov=[PosSymDefMatrix(H,i) for i in range(M)]
    broadness = 2

    mu=np.empty([M, H])
    sig = np.empty([M, H, H])
    
    for j in range(M):
        mu[j,:] = X[k[j]]
        sig[j,:,:] = np.mean(np.array(Cov[j]), axis=0)+broadness
    
    guess={'mu':mu,
          'sig':sig,
          'lambda':lam}


    shift = maxsize
    epsilon = 0.01
    iters = iterations
    df_copy = df.copy()

    df_copy['label']=map(lambda x:x+1, np.random.choice (M,len(df))) 
    params = pd.DataFrame.from_dict(guess, orient = 'index')
    params = params.transpose()
    

    while shift > epsilon:
        iters += 1
        updated_labels = expectation(df_copy.copy(), params, N, M, H)
        updated_parameters = maximization(updated_labels, params.copy(), N, M, H)
        shift = distance(params, updated_parameters, N, M, H)
        df_copy = updated_labels
        params = updated_parameters
    return np.array(df_copy.label)




def do_gmm(list_ensembles, nEnsCluster, iterations, verbose, N_clusters_max, hdf5_file_name):
    mixtureObj = emProcess(list_ensembles, nEnsCluster, iterations)
    return mixtureObj
