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
file_save_results='complete_results_batch2000'
file_save_clusters='cluster_user_batch_2000'


###  START DOING EXPERIMENTS  ###

scores_df=pd.DataFrame(columns=['method', 'clustering', 'ensemble', 'score', 'batch', 'cluster_num', 'database'])
cluster_user=pd.DataFrame(columns=['batch', 'cluster', 'clustering', 'user_id', 'ensemble', 'databse'])

for dataset in dataset_list:
    train, test, data=get_data.get_data_batch(dataset, batch)
    if dataset=='instacart':
        num_clusters=5
    else: 
        num_clusters=25
    print('RECEIVED DATASET')
    
    # NO CLUSTERS
    a=0
    for method in methods_list:
        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
        scores_df = scores_df.append(
            {'method': method_name[a], 'clustering': 'no_cluster', 'ensemble': 'no_ensemble', 'score': score, 
             'batch': batch, 'cluster_num': 'no_cluster', 'database':dataset},
            ignore_index=True)
        scores_df.to_csv(file_save_results)
        a=a+1
    
    print('FINISH METHODS FOR NO CLUSTERS')
    
    
    # SINGLE CLUSTER
    b=0
    for clustering in single_clustering_list:
        clustering_labels=clustering(train, test,data, num_clusters, batch, dataset)
        
        cluster_user=cluster_user.append({'batch': clustering_label.batch, 'cluster': clustering_label.cluster, 
                                                  'clustering': cluster_label.type_cluster, 'user_id': cluster_label.ix[:,0], 
                                                  'ensemble':'no_ensemble', 'databse': dataset})
        cluster_user.to_csv('file_save_clusters')
        test_b = test.merge(cluster_labels, on='user_id')
        train_b = train.merge(cluster_labels, on='user_id')
        data_b= data.merge(cluster_labels, on='user_id')
        cluster_list = train_b.cluster.unique()
        
        for cluster in cluster_list:
            data_cluster= data_b[data_b["cluster"]==cluster]
            train_cluster=train_b[train_b["cluster"]==cluster]
            test_cluster=test_b[test_b["cluster"]==cluster]
            c=0
            for method in methods_list:
                score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
                scores_df = scores_df.append(
            {'method': method_name[c], 'clustering': single_clustering_name[b], 'ensemble': 'no_ensemble', 'score': score, 
             'batch': batch, 'cluster_num': cluster, 'database':dataset},ignore_index=True)
                scores_df.to_csv(file_save_results)
                c=c+1
        b=b+1
        
    print('FINISH METHODS FOR SINGLE CLUSTER')
      
    #ENSEMBLE
    cluster_table=pd.pivot_table(clustering_labels, values='cluster', index=['user_id'], 
                                  columns=['type_cluster'], aggfunc=np.sum).dropna(how='any', axis=0)
    list_ensembles=np.append([np.array(cluster_table.cluster_features), np.array(cluster_table.cluster_kshape)], [np.array(cluster_table.cluster_graph)], axis=0)
    
    d=0
    for ensemble in ensemble_clustering_list:
        final_ensemble=ensemble
        final_ensemble_df=pd.DataFrame({'user_id':cluster_table.index, 'label':ensemble})
        cluster_user=cluster_user.append({'batch': batch, 'cluster': final_ensemble_df.label, 
                                                  'clustering': 'no_cluster', 'user_id': final_ensemble_df.ix[:,0], 
                                                  'ensemble':ensemble_clustering_name[d], 'databse': dataset})
        cluster_user.to_csv('file_save_clusters')
        data_d=data.merge(final_ensemble_df, on='user_id')
        train_d=train.merge(final_ensemble_df, on='user_id')
        test_d=test.merge(final_ensemble_df, on='user_id')
        cluster_list=train_d.cluster.unique()
        
        for cluster in cluster_list:
            data_cluster= data_d[data_d["cluster"]==cluster]
            train_cluster=train_d[train_d["cluster"]==cluster]
            test_cluster=test_d[test_d["cluster"]==cluster]
            e=0
            for method in methods_list:
                score=method(train_ensemble, test_ensemble, data_ensemble, return_pred, dataset)
                scores_df = scores_df.append(
            {'method': method_name[e], 'clustering': 'no_cluster', 'ensemble': ensemble_clustering_name[d], 'score': score, 
             'batch': batch, 'cluster_num': cluster, 'database':dataset},ignore_index=True)
                scores_df.to_csv(file_save_results)
                c=c+1
        b=b+1
    d=d+1
  print('FINISH METHODS FOR ENSEMBLE')
