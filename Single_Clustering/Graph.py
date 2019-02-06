import networkx as nx
from community import community_louvain
import pandas as pd
import numpy as np


"""Function to make clustering in data using Louvain algorithm
        Parameters:train, test,data, num_cluster, batch, dataset
        Return:  y_pred_test_df, pandas table with columns cluster, user_id, batch and type_cluster"""

def cluster_graph(train, test,data, num_cluster, batch, dataset):
    if dataset=='instacart':
        FG = nx.from_pandas_edgelist(train, source='user_id', target='product_name', edge_attr=True)
        parts = community_louvain.best_partition(FG)
        y_pred_df = pd.DataFrame.from_dict(parts, orient='index', columns=['cluster']).reset_index()
        y_pred_df=y_pred_df.rename({'index': 'user_id'}, axis='columns')
        y_pred_df['user_id'] = pd.to_numeric(y_pred_df['user_id'], errors='coerce')
        y_pred_df=y_pred_df.dropna()
        y_pred_df['batch']=batch
        y_pred_df['type_cluster']='cluster_graph'
        #y_pred_df.to_csv('clusters_graph.csv')
    
    else:
        FG = nx.from_pandas_edgelist(data, source='card_id', target='merchant_id', edge_attr=True)
        parts = community_louvain.best_partition(FG)
        y_pred_df = pd.DataFrame.from_dict(parts, orient='index', columns=['cluster']).reset_index()
        y_pred_df=y_pred_df.rename({'index': 'card_id'}, axis='columns')
        y_pred_df['batch']=batch
        y_pred_df['type_cluster']='cluster_graph'
    return y_pred_df
