### IMPORT LIBRARIES AND FILES  ###
import numpy as np
import pandas as pd
from tqdm import tqdm
import get_data#.get_data_batch (dataset_name, batch)
import Features#.cluster_features(train, test, data, num_cluster, batch, dataset)
import Graph#.cluster_graph(train, test,data, num_cluster, batch, dataset)
import Time_Series#.cluster_kshape(train, test,data, num_cluster, batch, dataset)
import graph_ensemble #do_graph(list_ensembles, verbose = True, N_clusters_max = 5)
import Mixture_Models#.do_mixturemodels(list_ensembles, nEnsCluster=5, iterations=10)
import Voting#.do_voting(list_ensembles)
import Catboost_nba#do_catboost(train, test,data, return_pred, dataset)
import Lightfm_nba #do_lightfm(train, test, data, return_pred, dataset)
import easiest_nba #easiest(train, test, data, return_pred, dataset)
import xgboost_nba #do_xgboost(train, test, data, return_pred, dataset)
import sys
sys.path.insert(0, '/the/folder/path/name-folder/')

###  DEFINE PARAMETERS  ###
dataset_list=['instacart', 'elo']

single_clustering_list=[Features.cluster_features,Graph.cluster_graph, Time_Series.cluster_kshape]
single_clustering_name=['cluster_features','cluster_graph', 'cluster_timeseries']

ensemble_clustering_list=[graph_ensemble.do_graph(list_ensembles, verbose = True, N_clusters_max = num_clusters), 
                          Mixture_Models.do_mixturemodels(list_ensembles, nEnsCluster=5, iterations=10), 
                          Voting.do_voting(list_ensembles)]
ensemble_clustering_name=['graph_ensemble', 'Mixture_Models', 'Voting']


methods_list = [easiest_nba.easiest, Lightfm_nba.do_lightfm, xgboost_nba.do_xgboost, Catboost_nba.do_catboost]
methods_name=['simplest','lightfm','xgboost','catboost']

return_pred=0
batch='users_2000'



###  START DOING EXPERIMENTS  ###

scores_df=pd.DataFrame(columns=['method', 'cluster', 'ensemble', 'score', 'batch', 'cluster', 'database'])

for dataset in dataset_list:
    train, test, data=get_data.get_data_batch(dataset, batch)
    if dataset=='instacart':
        num_clusters=5
    else: 
        num_clusters=25
    print('RECEIVED DATASET')
    
    # NO CLUSTERS
    for method in methods_list:
        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
        scores_batch = scores_batch.append(
            {'batch': batch, 'method': methods_name[i], 'recall': recall, 'precision': precision, 'fscore': fscore},
            ignore_index=True)
        scores_batch.to_csv('scores_batch_xboost_clustering.csv')
    
    print('FINISH METHODS FOR NO CLUSTERS')

    for clustering in single_clustering_list:
        cluster=clustering(train, test,data, num_clusters, batch, dataset)
        data_cluster= data[data["cluster"]==cluster]
        train_cluster=train[train["cluster"]==cluster]
        test_cluster=test[test["cluster"]==cluster]
        
        # SINGLE CLUSTER
        for method in methods_list:
            score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
            
    list_ensembles=[]
    for ensemble in ensemble_clustering_list:
        final_ensemble=ensemble
        data_ensemble=
        train_ensemble=
        test_ensemble=
        
        # ENSEMBLE
        for method in methods_list:
            score=method(train_ensemble, test_ensemble, data_ensemble, return_pred, dataset)
            
