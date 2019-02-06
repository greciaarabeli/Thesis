""" https://github.com/NaegleLab/OpenEnsembles/tree/master/openensembles """

import pandas as pd
import numpy as np
from functools import reduce
from six.moves import reduce


def emProcess(list_ensembles,N , nEnsCluster, iterations):
    y = gatherPartitions(list_ensembles, N)
    K = genKj(y)
    alpha, v, ExpZ = initParameters(nEnsCluster, y, K)
    def piConsensus(ExpZ):
        '''
            The function outputs the final ensemble solution based on ExpZ values.
            '''
        maxExpZValues = {}
        piFinishing = {}
        labels = []
        
        for i in range(ExpZ.shape[0]):
            maxExpZValues[i] = max(ExpZ[i,:]) 
            piFinishing[i] = []

            for j in range(ExpZ.shape[1]):
                if (maxExpZValues[i] == ExpZ[i,j]):
                    piFinishing[i].append(j + 1)
                    labels.append(j+1)

        # choose randomly in the case of same values of ExpZ[i,:]          
        #[piFinishing[i].delete(random.choice(piFinishing[i])) for i in piFinishing.keys() if (len(piFinishing[i]) > 1)]             
        return piFinishing, labels
        
    i = 0
    while(i<iterations):
        ExpZ = expectation(ExpZ, y, K, v, alpha)
        alpha, v = maximization(alpha, ExpZ, y, K, v)
        i += 1

    piFinishing, labels = piConsensus(ExpZ)
    piFinishing = piFinishing
    labels = np.asarray(labels)
    #return piFinishing, labels
    return labels
    
    
def gatherPartitions(list_ensembles, N):
    '''
    Returns the y vector.
    parg: list of H-labeling solutions
    nElem: number of features/objects
    '''
    H = len(list_ensembles)
    listParts = np.concatenate(list_ensembles).reshape(H,N)
    #print listParts[:,0]

    y = [] 
    [y.append(listParts[:,i]) for i in range(N)]

    y = pd.DataFrame(y, columns= np.arange(H))
    y.index.name = 'objs'
    y.columns.name = 'partition'
    return y
    
    
ef genKj(y):
    '''
    Generates the K(j) H-array that contains the tuples of unique 
    clusters of each j-th partition, eg: K = [(X,Y), (A,B)] 
    '''
    #K = np.zeros(y.shape[1], dtype= int)
    K = []
    aux = []
    for i in range(y.shape[1]):
        if 'NaN' in np.unique(y.iloc[:,i].values):
            aux = copy.copy(y.iloc[:,i].values)
            aux = [x for x in aux if x != 'NaN']
            K.append(aux)
        else:
            K.append(tuple(np.unique(y.iloc[:,i].values)))
    return K
    
    
    
 def initParameters(nEnsCluster, y, K):
        '''
        The function initializes the parameters of the mixture model.
        '''    
        def initAlpha(nEnsCluster):
            return np.ones(nEnsCluster) / nEnsCluster
            
        def initV(nEnsCluster, y, K):
            v = []
            [v.append([]) for j in range(y.shape[1])]
            
            #[v[j].append(list(np.ones(len(self.K[j])) / len(self.K[j]))) for j in range(self.y.shape[1]) for m in range(self.nEnsCluster)]
            for j in range(y.shape[1]):
                for m in range(nEnsCluster):
                    aux = abs(np.random.randn(len(K[j])))
                    v[j].append( aux / sum(aux) )
        
            return v
    
        def initExpZ(nEnsCluster, y):
            return np.zeros(y.shape[0] * nEnsCluster).reshape(y.shape[0],nEnsCluster)
    
        alpha = initAlpha(nEnsCluster)
        v = initV(nEnsCluster,y, K)
        ExpZ = initExpZ(nEnsCluster, y)
        return alpha, v, ExpZ
        
 def expectation(ExpZ, y, K, v, alpha):
    '''
    Compute the Expectation (ExpZ) according to parameters.
    Obs: y(N,H) Kj(H) alpha(M) v(H,M,K(j)) ExpZ(N,M)
    '''
    def sigma(a,b):
        return 1 if a == b else 0

    M = ExpZ.shape[1]
    nElem = y.shape[0]


    for m in range(M):
        for i in range(nElem):

            prod1 = 1
            num = 0
            for j in range(y.shape[1]):
                ix1 = 0
                for k in K[j]:
                    prod1 *= ( v[j][m][ix1] ** sigma(y.iloc[i][j],k) )
                    ix1 += 1
            num += alpha[m] * prod1

            den = 0
            for n in range(M):

                prod2 = 1
                for j2 in range(y.shape[1]):
                    ix2 = 0
                    for k in K[j2]:
                        prod2 *= ( v[j2][n][ix2] ** sigma(y.iloc[i][j2],k) )
                        ix2 += 1
                den += alpha[n] * prod2


            ExpZ[i][m] = float(num) / float(den)

    return ExpZ
    
    
 def maximization(alpha, ExpZ, y, K, v):
    '''
    Update the parameters taking into account the ExpZ computed in the 
    Expectation (ExpZ) step.
    Obs: y(N,H) Kj(H) alpha(M) v(H,M,K(j)) ExpZ(N,M)
    '''
    def vecSigma(vec, k):
        '''
        Compare i-th elements of vector to k assigining to 
        a vector 1 if i-th == k, 0 otherwise. 
        '''
        aux = []
        for i in vec:
            if i == k:
                aux.append(1)
            else:
                aux.append(0)
        return np.asarray(aux)
    
    def updateAlpha(alpha, ExpZ):
        for m in range(alpha.shape[0]):
            alpha[m] = float(sum(ExpZ[:,m])) / float(sum(sum(ExpZ)))
        return alpha
    
    def updateV(y, alpha, K, ExpZ, v):
        for j in range(y.shape[1]):
            for m in range(alpha.shape[0]):
                ix = 0
                for k in K[j]:
                    num = sum(vecSigma(y.iloc[:,j],k) * np.array(ExpZ[:,m]))
                    den = 0
                    for kx in K[j]:
                        den += sum(vecSigma(y.iloc[:,j],kx) * np.asarray(ExpZ[:,m]))
                    v[j][m][ix] = float(num) / float(den)
                    ix += 1
                            
        return v
                
    alpha = updateAlpha(alpha, ExpZ )
    v = updateV(y, alpha, K, ExpZ, v)
    return alpha, v
    
   
"""
Finishing Technique to assemble a final, hard parition of the data according to maximizing the likelihood according to the
observed clustering solutions across the ensemble. This will operate on all clustering solutions contained in the container cluster class.
Operates on entire ensemble of clustering solutions in self, to create a mixture model
See finishing.mixture_model for more details. 
Parameters

References
Topchy, Jain, and Punch, "A mixture model for clustering ensembles Proc. SIAM Int. Conf. Data Mining (2004)"
"""

def do_mixturemodels(list_ensembles, nEnsCluster=5, iterations=10):
  mixtureObj = emProcess(list_ensembles,list_ensembles.shape[1] , nEnsCluster, iterations)
  return mixtureObj
