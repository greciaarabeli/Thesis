from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

"""Function to make clustering in data using Kshape
        Parameters:train, test,data, num_cluster, batch, dataset
        Return:  y_pred_test_df, pandas table with columns cluster, user_id, batch and type_cluster"""

def cluster_features(train, test, data, num_cluster, batch, dataset):
    
    random_state=2
    if dataset =='instacart':
        user_features=['user_id','total', 'total_reorder', 'Totalmin', 'Totalmax',
       'Totalmean', 'order_numbermax', 'days_since_prior_ordermin',
       'days_since_prior_ordermax', 'days_since_prior_ordermean',
       'reordermin', 'reordermax', 'reordermean',
       'order_hour_of_daymin', 'order_hour_of_daymax', 'order_hour_of_daymean',
       'days_since_prior_ordermin', 'days_since_prior_ordermax',
       'days_since_prior_ordermean', 'order_dowmean']

        train=train.drop_duplicates(subset='user_id', keep='first')
        train1=train[user_features]
        y_pred_train = KMeans(n_clusters=num_cluster, random_state=random_state).fit_predict(train1)
        y_pred_train_df = pd.DataFrame(y_pred_train)
        y_pred_train_df['user_id'] = train.user_id.values
        y_pred_train_df= y_pred_train_df.rename({0: 'cluster'}, axis='columns')
        y_pred_train_df['batch'] = batch
        y_pred_train_df['type_cluster']='cluster_features'
        
        return y_pred_train_df

    else:
        train_test=train.append(test)
        train_test1=train_test.drop(['card_id', 'first_active_month', 'target', 'batch'], axis=1)
        y_pred_test = KMeans(n_clusters=num_cluster, random_state=random_state).fit_predict(train_test1)
        y_pred_test_df = pd.DataFrame(y_pred_test)
        y_pred_test_df['card_id'] = train_test.card_id.values
        y_pred_test_df= y_pred_test_df.rename({0: 'cluster'}, axis='columns')
        y_pred_test_df['batch'] = batch
        y_pred_test_df['type_cluster']='cluster_features'
    
        return y_pred_test_df
