import numpy as np
import pandas as pd
from tqdm import tqdm
import get_data#.get_data_batch (dataset_name, batch)
import Features#.cluster_features(train, test, data, num_cluster, batch, dataset)
import Graph#.cluster_graph(train, test,data, return_pred, num_cluster, batch, dataset)
import Time_Series#.cluster_kshape(train, test,data, num_cluster, batch, dataset)
import graph_ensemble #do_graph(list_ensembles, verbose = True, N_clusters_max = 5)
import Mixture_Models#.do_mixturemodels(list_ensembles, nEnsCluster=5, iterations=10)
import Voting#.do_voting(list_ensembles)
import Catboost_nba#do_catboost(train, test,data, return_pred, dataset)
import Lightfm_nba #do_lightfm(train, test, data, return_pred, dataset)
import easiest_nba #easiest(train, test, data, return_pred, dataset)
import xgboost_nba #do_xgboost(train, test, data, return_pred, dataset)

dataset=['instacart', 'elo']

single_clustering_list=[Features.cluster_features,Graph.cluster_graph, Time_Series.cluster_kshape]
single_clustering_name=['cluster_features','cluster_graph', 'cluster_timeseries']

ensemble_clustering_list=[graph_ensemble.do_graph, Mixture_Models.do_mixturemodels, Voting.do_voting]
ensemble_clustering_name=['graph_ensemble', 'Mixture_Models', 'Voting']


methods_list = [easiest_nba.easiest, Lightfm_nba.do_lightfm, xgboost_nba.do_xgboost, Catboost_nba.do_catboost]
methods_name=['simplest','lightfm','xgboost','catboost']



batches=test.batch.unique()

scores_batch=pd.DataFrame(columns=['batch', 'method', 'recall', 'precision', 'fscore'])
num_cluster=10
return_pred=0
for batch in batches:
    i = 0
    train_batch=train[train["batch"]==batch]
    test_batch=test[test["batch"]==batch]
    for method in methods_list:
        recall, precision, fscore=method(train_batch, test_batch, return_pred, num_cluster) #1 if yes 0 if not
        scores_batch = scores_batch.append(
            {'batch': batch, 'method': methods_name[i], 'recall': recall, 'precision': precision, 'fscore': fscore},
            ignore_index=True)
        scores_batch.to_csv('scores_batch_xboost_clustering.csv')
        i = i + 1

scores_batch.to_csv('scores_batch_xboost_clustering.csv')
