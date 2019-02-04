from tslearn.utils import to_time_series_dataset
import tslearn
import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt


def normalize_data(data):
    X_train = TimeSeriesScalerMeanVariance().fit_transform(data)
    sz = X_train.shape[1]
    return X_train, sz

def k_shape(X_train, n_clusters, verbose=True, seed=0):
    # Euclidean k-means
    ks = KShape(n_clusters=n_clusters, verbose=verbose, random_state=seed)

    return ks, ks.fit_predict(X_train)

def compute_scores(ks, X_train, y_pred, centroid=False):
    scores = []
    # range(ks.n_clusters)
    for yi in np.unique(y_pred):
        tp_list = []
        for xx in X_train[y_pred == yi]:

            if centroid:
                predicted = ks.cluster_centers_[yi].ravel()
                actual = xx.ravel()
                score = adjusted_rand_score(actual, predicted)
                scores.append(score)

            else:
                predicted = xx.ravel()
                tp_list.append(predicted)

        if not centroid:
            half = len(tp_list) // 2
            first_half = tp_list[:half]
            second_half = tp_list[half:]

            for i in np.arange(half):
                score = adjusted_rand_score(first_half[i], second_half[i])
                scores.append(score)

    return scores

def plot_data(ks, X_train, y_pred, sz, n_clusters=3, centroid=False):
    plt.figure(figsize=(12, 25))
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        for xx in X_train[y_pred == yi]:
            # , alpha=.2
            plt.plot(xx.ravel(), "k-")
            # ,
        if centroid:
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        # plt.ylim(-4, 4)
        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()


def make_timeseries(train, dataset):
    if dataset == 'instacart':
        orders = train.drop_duplicates(subset=['order_id', 'user_id'], keep='first').sort_values('order_number')
        orders['week'] = (orders.groupby(['user_id'])['days_since_prior_order'].cumsum() / 7).fillna(0).astype(int)
        orders['week_day'] = orders[['week', 'order_dow']].values.tolist()
        cross_ts = pd.crosstab(orders.user_id, orders.week, values=orders.total, aggfunc='sum')
        cross_ts = cross_ts.fillna(0).values.tolist()
        cross_ts1=cross_ts
    
    else:
        cross_ts = pd.crosstab(train.card_id, train.year_month_purch, values=train.purchase_amount, aggfunc='sum')
        cross_ts1 = cross_ts.fillna(0).values.tolist()
    return cross_ts1, cross_ts.index
    

"""" Function to  perform Kshape clustering
    Parameters:
            train
            test
            data
            num_cluster
            batch
            dataset
    
    Return:
            y_pred_df: pandas table with columns cluster, user_id, batch and type_cluster
"""""
def cluster_kshape(train, test,data, num_cluster, batch, dataset):
    if dataset == 'instacart':
        ts, ts_index=make_timeseries(train, dataset)

        formatted_dataset = to_time_series_dataset(ts)
        X_train, sz = normalize_data(formatted_dataset)
        ks, y_pred = k_shape(X_train, n_clusters=num_cluster)
        scores = compute_scores(ks, X_train, y_pred)
        plt.boxplot(scores)
        silhouette= tslearn.clustering.silhouette_score(X_train, y_pred, metric="euclidean")
        print("For n_clusters =", num_cluster,
                "The average silhouette_score is :", silhouette)
        plot_data(ks, X_train, y_pred, sz, ks.n_clusters, centroid=True)
        y_pred_df = pd.DataFrame(y_pred)
        userindex = train.user_id.unique()
        userindex = np.sort(userindex)
        y_pred_df['user_id'] = userindex
        y_pred_df= y_pred_df.rename({0: 'cluster'}, axis='columns')
        y_pred_df['batch'] = batch
        y_pred_df['type_cluster']='cluster_kshape'

    

    else:
        ts, ts_index=make_timeseries(data, dataset)
        sum_pred_test=pd.DataFrame()
        formatted_dataset = to_time_series_dataset(ts)
        X_train, sz = normalize_data(formatted_dataset)
        ks, y_pred = k_shape(X_train, n_clusters=num_cluster)
        scores = compute_scores(ks, X_train, y_pred)
        plt.boxplot(scores)

        plot_data(ks, X_train, y_pred, sz, ks.n_clusters, centroid=True)
        y_pred_df = pd.DataFrame(y_pred)
        y_pred_df['card_id'] = ts_index
        y_pred_df= y_pred_df.rename({0: 'cluster'}, axis='columns')
        y_pred_df['batch'] = batch
        y_pred_df['type_cluster']='cluster_kshape'
    return y_pred_df
