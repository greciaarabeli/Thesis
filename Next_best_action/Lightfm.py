from lightfm import LightFM
import numpy as np
import pandas as pd

def lightfm(train, test, data, return_pred, dataset):
    if dataset=='instacart':
        set1 = set(np.unique(train.product_id))
        set2 = set(np.unique(test.product_id))
        missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
        added = pd.DataFrame.from_dict(list(sorted(set2 - set1)))

        for i in range(len(missing)):
            a = missing[0][i]
            test = test.append({'product_id': a}, ignore_index=True)
        for i in range(len(added)):
            a = added[0][i]
            train = train.append({'product_id': a}, ignore_index=True)

        train = train.fillna(0)
        test = test.fillna(0)

        grouped_train_i = train.groupby(["user_id", "product_id"])["reordered"].aggregate("sum").reset_index()
        grouped_test_i = test.groupby(["user_id", "product_id"])["reordered"].aggregate("sum").reset_index()

        interactions_i = lightfm_form.create_interaction_matrix(df=grouped_train_i,
                                                                user_col='user_id',
                                                                item_col='product_id',
                                                                rating_col='reordered')

        interactions_test_i = lightfm_form.create_interaction_matrix(df=grouped_test_i,
                                                                     user_col='user_id',
                                                                     item_col='product_id',
                                                                     rating_col='reordered')

        mf_model = lightfm_form.runMF(interactions=interactions_i,
                                      n_components=30, loss='warp', epoch=40, n_jobs=4)

        test_history = test[test['reordered'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
        test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
        n_users, n_items = interactions_i.shape

        results = []
        test_history['pred'] = 0
        for user_id in test_history['user_id']:
            print(user_id)
            recom = mf_model.predict(user_id, np.arange(n_items), num_threads=4)
            recom = pd.Series(recom)
            recom.sort_values(ascending=False, inplace=True)
            if (len(results) == 0):
                results = np.array(recom.iloc[0:10].index.values)
            else:
                results = np.vstack((results, recom.iloc[0:10].index.values))

        results_df = pd.DataFrame(data=results)
        columns = results_df.columns.values
        test_history['pred'] = results_df[columns].values.tolist()

        y_pred = test_history['pred']
        y_true = test_history['product_id']

        test_precision = precision_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
        test_recall = recall_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
        f_test = 2 * test_precision * test_recall / (test_precision + test_recall)

        if return_pred==0:
            return f_test
        else:
            return f_test, y_pred
    
    
    else:
        train=train.merge(data, on='card_id', how='left')
        train_avgtarget = train.groupby(["merchant_id"])["target"].aggregate("mean").reset_index()
        train=train.merge(train_avgtarget, on='merchant_id', how='left')
        test=test.merge(data, on='card_id', how='left')
        test=test.merge(train_avgtarget, on='merchant_id', how='left')
        test_train=train.append(test)
        grouped_train_test = test_train.groupby(["merchant_id", "card_id"])["target_y"].aggregate("mean").reset_index()
        interactions = lightfm_form.create_interaction_matrix(df=grouped_train_test,user_col='merchant_id',item_col='card_id',rating_col='target_y')
        train_unique=train.drop_duplicates(subset=['card_id'])
        test_unique=test.drop_duplicates(subset=['card_id'])
        item_features= train_unique.append(test_unique)[['feature_1', 'feature_2', 'feature_3',
           'num_transactions', 'sum_trans', 'mean_trans', 'std_trans', 'min_trans',
           'max_trans', 'year_first', 'month_first']]
        mf_model = lightfm_form.runMF(interactions=interactions,
                                      n_components=30, loss='warp', epoch=40, n_jobs=4)
        # Create User Dict
        user_dict = create_user_dict(interactions=interactions)
        # Create Item dict
        products_dict = create_item_dict(df = data.reset_index(),
                                   id_col = 'card_id',
                                   name_col = 'card_id')
        ## Creating item-item distance matrix
        item_item_dist = create_item_emdedding_distance_matrix(model = mf_model,
                                                           interactions = interactions)

        scores_rmse=pd.DataFrame(columns=['card_id', 'pred'])

        for cards in test.card_id.unique():
            recommended_items = list(pd.Series(item_item_dist.loc[cards,:]. \
                                      sort_values(ascending = False).head(10+1). \
                                      index[1:10+1]))
            recommended_train=list(train_unique[train_unique.card_id.isin(recommended_items)].card_id.values)
            pred=train_unique.loc[train_unique['card_id'].isin(recommended_train)]
            scores_rmse=scores_rmse.append(
                {'card_id': cards, 'pred': pred.target_x.mean()},ignore_index=True)
        scores_rmse=scores_rmse.merge(test_unique[['card_id', 'target_x']], on='card_id')
        scores_rmse=scores_rmse.fillna(scores_rmse.pred.mean())
        if return_pred==0:
            return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x)), scores_rmse
        else:
            return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x)), scores_rmse
    
    
    
    print('FINISH LIGHTFM')
