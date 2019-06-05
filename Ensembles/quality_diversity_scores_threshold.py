from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random as rand
from sys import maxsize
import prince


### DAVIES-BOULDIN ###
def db_index(data, labels):
    db = davies_bouldin_score(data, labels)
    
    #FOR INDIEX WHERE THE SMALLEST THE BETTER WE SHOULD RECALCULATE IT USING EXP NEGATIVE
    return np.exp(-1*db)




def inter_dist(ci, cj, distance):
    values = distance[np.where(ci)][:, np.where(cj)]
    values = values[np.nonzero(values)]
    return np.min(values)
    
    
    
def intra_dist(ci, distance):
    values = distance[np.where(ci)][:, np.where(ci)]
    return np.max(values)
    
### DUNN INDEX ###    
def dunn_index(data, labels):
    """
    Parameters:
        labels: np.array[N] cluster labels of all points
        
        data: np.array([N, f]) of all points where f its the number of features
    
    """
    k_list=np.unique(labels)
    k=len(k_list)
    k_range=list(range(0, k))
    matrix_distance=euclidean_distances(data)
    
    min_dist=np.ones([k, k])
    max_dist=np.ones([k,1])
    
    for i in k_range:
        for j in (k_range[0:i]+k_range[i+1:]):
            min_dist[i,j]=inter_dist(labels==k_list[i], labels==k_list[j],matrix_distance)
        max_dist[i]=intra_dist(labels==k_list[i], matrix_distance)
    d=np.min(min_dist)/np.max(max_dist)
    
    return d

### CALINSKI - HARABASZ ###
def ch_score(data, labels):
    s=calinski_harabaz_score(data, labels)
    return s

### QUALITY ###
def do_quality(data, c_list):
    list_validityindex=[db_index, dunn_index, silhouette_score, ch_score]
    civi_i=[]
    for j in list_validityindex:
        civi_list_i=[]
        for i in range(len(c_list)):
            civi=j(data, c_list[i])
            civi_list_i.append(civi)
        sum_civi_i=np.sum(civi_list_i)
        #normalize the civi
        if sum_civi_i == 0:
            civi_i.append(np.zeros(len(c_list)))
        else:
            civi_i.append(np.array(civi_list_i/sum_civi_i))

    #CALCUALTE QUALITY
    quality=[]
    for i in range(len(c_list)):
        sum_civi=0
        for j in range(len(list_validityindex)):
            sum_civi=sum_civi+civi_i[j][i]
        quality_i=(1/len(list_validityindex))*sum_civi
        quality.append(quality_i)
     
    return quality

### DIVERSITY ###
def do_diversity(data, c_list):
    diversity=[]
    for i in range(len(c_list)):
        sum_nmi=0
        for j in range(len(c_list)):
            if i!=j:
                nmi=normalized_mutual_info_score(c_list[i], c_list[j])
                sum_nmi=sum_nmi+(1-nmi)
            else:
                sum_nmi=sum_nmi
        diversity_i=sum_nmi/(len(c_list)-1)
        diversity.append(diversity_i)
    
    return diversity


### JOINT QUALITY AND DIVERSITY ###
def joint_qual_diver(data, c_list, alpha):
    quality=do_quality(data, c_list)
    diversity=do_diversity(data, c_list)
    final_score=(alpha*np.array(quality))+((1-alpha)*np.array(diversity))

    return final_score

""" Return the elements of the ensemble that has join quality and diversity score above mean"""
#alpha is the weight assigned to quality
#threshold is the minimum score that eanch clustering base should have to be part of the ensemble
def ensembles_above_threshold(list_ensembles, data,alpha, threshold):
    final_score= joint_qual_diver(data, list_ensembles, alpha)
    #let just partitions with final score higher than the mean
    if threshold=='mean':
        index_mean=np.where(final_score >= np.mean(final_score))
        fn=list_ensembles[index_mean]
    else:
        index_threshold=np.where(final_score >= threshold)
        fn=list_ensembles[index_threshold]
    return fn
    
    
