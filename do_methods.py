#### IMPORT LIBRARIES AND FILES  ###
import numpy as np
import pandas as pd
import sys

from Data import get_data   #.get_data_batch (dataset_name, batch)
from Single_Clustering import Features  #.cluster_features(train, test, data, num_cluster, batch, dataset)
from Single_Clustering import Graph#.cluster_graph(train, test,data, num_cluster, batch, dataset)
from Single_Clustering import Time_Series#.cluster_kshape(train, test,data, num_cluster, batch, dataset)

from Ensembles import Graph_ensemble #do_graph(list_ensembles, nEnsCluster=5, iterations=10, verbose = True, N_clusters_max = 5, hdf5_file_name=None)
from Ensembles import Mixture_Models#.do_mixturemodels(list_ensembles, nEnsCluster=5, iterations=10, verbose = True, N_clusters_max = 5, hdf5_file_name=None)
from Ensembles import Voting#.do_voting(list_ensembles, nEnsCluster=5, iterations=10, verbose = True, N_clusters_max = 5, hdf5_file_name=None

from Next_best_action import Catboost_nba#do_catboost(train, test,data, return_pred, dataset)
from Next_best_action import Lightfm_nba #do_lightfm(train, test, data, return_pred, dataset)
from Next_best_action import easiest_nba #easiest(train, test, data, return_pred, dataset)
from Next_best_action import xgboost_nba #do_xgboost(train, test, data, return_pred, dataset)



###  DEFINE PARAMETERS  ###
dataset_list=['instacart', 'elo']

single_clustering_list=[Features.cluster_features,Graph.cluster_graph, Time_Series.cluster_kshape]
single_clustering_name=['cluster_features','cluster_graph', 'cluster_timeseries']

ensemble_clustering_list=[Graph_ensemble.do_graph, Mixture_Models.do_mixturemodels,
                          Voting.do_voting]
ensemble_clustering_name=['Graph_ensemble', 'Mixture_Models', 'Voting']


methods_list = [easiest_nba.easiest, Lightfm_nba.do_lightfm, xgboost_nba.do_xgboost, Catboost_nba.do_catboost]
methods_name=['simplest','lightfm','xgboost','catboost']

return_pred=0
batch='users_2000'
file_save_results='complete_results_batch2000_complete.csv'
file_save_clusters='cluster_user_batch_2000_complete.csv'


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
        score=method(train, test, data, return_pred, dataset)
        scores_df = scores_df.append(
            {'method': methods_name[a], 'clustering': 'no_cluster', 'ensemble': 'no_ensemble', 'score': score, 
             'batch': batch, 'cluster_num': 'no_cluster', 'database':dataset},
            ignore_index=True)
        scores_df.to_csv(file_save_results)
        a=a+1
    
    print('FINISH METHODS FOR NO CLUSTERS')
     
    
    # SINGLE CLUSTER
    clustering_labels=pd.DataFrame()
        
    b=0
    for clustering in single_clustering_list:
        clustering_labels_b=clustering(train, test,data, num_clusters, batch, dataset)
        
        
        if dataset=='instacart':
            clustering_labels_b['ensemble']='no_enemble'
            clustering_labels_b['database']='instacart'
            clustering_labels_b['card_id']=clustering_labels_b['user_id']
            clustering_labels=clustering_labels.append(clustering_labels_b)

            test_b = test.merge(clustering_labels_b, on='user_id')
            train_b = train.merge(clustering_labels_b, on='user_id')
            data_b=data
            
        else:
            clustering_labels_b['ensemble']='no_enemble'
            clustering_labels_b['database']='elo'
            clustering_labels_b['user_id']=clustering_labels_b['card_id']
            clustering_labels=clustering_labels.append(clustering_labels_b)
            

            test_b = test.merge(clustering_labels_b, on='card_id')
            train_b = train.merge(clustering_labels_b, on='card_id')
            data_b= data.merge(clustering_labels_b, on='card_id')
        
        clustering_labels.to_csv(file_save_clusters)
        
        cluster_list = train_b.cluster.unique()
        
        for cluster in cluster_list:
            try:
                data_cluster= data_b[data_b["cluster"]==cluster]
            except:
                data_cluster=data
            train_cluster=train_b[train_b["cluster"]==cluster]
            test_cluster=test_b[test_b["cluster"]==cluster]
            c=0
            for method in methods_list:
                if dataset == 'instacart':
                    if len(test_cluster.user_id.unique()) >1 and len(train_cluster.user_id.unique()) >1:
                    #try:
                        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
                    else:
                        score=np.nan
                        #except:
                            #pass
                else:
                    if len(test_cluster.card_id.unique()) >1 and len(train_cluster.card_id.unique()) >1:
                    #try:
                        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
                    else:
                        score=np.nan
                        #except:
                            #pass
                scores_df = scores_df.append(
                    {'method': methods_name[c], 'clustering': single_clustering_name[b], 'ensemble': 'no_ensemble', 'score':                                  score, 'batch': batch, 'cluster_num': cluster, 'database':dataset},ignore_index=True)
                scores_df.to_csv(file_save_results)
                    
                c=c+1
        b=b+1
        
    print('FINISH METHODS FOR SINGLE CLUSTER')
    
    #ENSEMBLE
    cluster_table=pd.pivot_table(clustering_labels, values='cluster', index=['user_id'], 
                                  columns=['type_cluster'], aggfunc=np.sum).dropna(how='any', axis=0)
    list_ensembles=np.append([np.array(cluster_table.cluster_features), np.array(cluster_table.cluster_kshape)],     [np.array(cluster_table.cluster_graph)], axis=0)

    d=0
    for ensemble in ensemble_clustering_list:
        final_ensemble=ensemble(list_ensembles, nEnsCluster=num_clusters, iterations=10, verbose = True, N_clusters_max =num_clusters , hdf5_file_name=None)
        final_ensemble_df=pd.DataFrame({'user_id':cluster_table.index, 'cluster':final_ensemble})
        if dataset=='instacart':
            final_ensemble_df['ensemble']=ensemble_clustering_name[d]
            final_ensemble_df['database']='instacart'
            final_ensemble_df['card_id']=final_ensemble_df['user_id']
            final_ensemble_df['batch']=batch
            final_ensemble_df['clustering']='no_cluster'
            clustering_labels=clustering_labels.append(final_ensemble_df)

            test_d = test.merge(final_ensemble_df, on='user_id')
            train_d = train.merge(final_ensemble_df, on='user_id')
            data_d=data
            
        else:
            final_ensemble_df['ensemble']=ensemble_clustering_name[d]
            final_ensemble_df['database']='elo'
            final_ensemble_df['card_id']=final_ensemble_df['user_id']
            final_ensemble_df['batch']=batch
            final_ensemble_df['clustering']='no_cluster'
            clustering_labels=clustering_labels.append(final_ensemble_df)
            

            test_d = test.merge(final_ensemble_df, on='card_id')
            train_d = train.merge(final_ensemble_df, on='card_id')
            data_d= data.merge(final_ensemble_df, on='card_id')
        
        clustering_labels.to_csv(file_save_clusters)
        
        cluster_list = train_d.cluster.unique()

        for cluster in cluster_list:
            try:
                data_cluster= data_d[data_d["cluster"]==cluster]
            except:
                data_cluster=data_d
            train_cluster=train_d[train_d["cluster"]==cluster]
            test_cluster=test_d[test_d["cluster"]==cluster]
            e=0
            for method in methods_list:
                if dataset == 'instacart':
                    if len(test_cluster.user_id.unique()) >1 and len(train_cluster.user_id.unique()) >1:
                    #try:
                        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
                    else:
                        score=np.nan
                        #except:
                            #pass
                else:
                    if len(test_cluster.card_id.unique()) >1 and len(train_cluster.card_id.unique()) >1:
                    #try:
                        score=method(train_cluster, test_cluster, data_cluster, return_pred, dataset)
                    else:
                        score=np.nan
                        #except:
                            #pass
                scores_df = scores_df.append(
                    {'method': methods_name[e], 'clustering': 'no_cluster', 'ensemble': ensemble_clustering_name[d], 'score':                                  score, 'batch': batch, 'cluster_num': cluster, 'database':dataset},ignore_index=True)
                scores_df.to_csv(file_save_results)

                e=e+1
        d=d+1
    print('FINISH METHODS FOR ENSEMBLE')
