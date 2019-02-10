""" Function to calculate the next best action according to each data set using Lightfm recomendation system.
    Parameters:train
               test
               data
               return_pred: 0 if you just need the accuracy score and 1 if you need the prediction
               dataset: 'instacart' or 'elo'
               
   Return: if 0 return average f1 for instacart and average MSE for elo
           if 1 return the same as 0 plus the predictions for test
"""


from lightfm import LightFM
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input -
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict



def create_item_emdedding_distance_matrix(model,interactions):
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix

def runMF(interactions, n_components, loss, epoch,n_jobs):
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,learning_schedule='adagrad')
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input -
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output -
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

def do_lightfm(train, test, data, return_pred, dataset):
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

        interactions_i = create_interaction_matrix(df=grouped_train_i,
                                                                user_col='user_id',
                                                                item_col='product_id',
                                                                rating_col='reordered')

        interactions_test_i = create_interaction_matrix(df=grouped_test_i,
                                                                     user_col='user_id',
                                                                     item_col='product_id',
                                                                     rating_col='reordered')

        mf_model = runMF(interactions=interactions_i,
                                      n_components=30, loss='warp', epoch=40, n_jobs=4)

        test_history = test[test['reordered'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
        test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
        n_users, n_items = interactions_i.shape
        
        user_dict = create_user_dict(interactions=interactions_i)
   
        print(interactions_i.shape)
        print(interactions_test_i.shape)
        print(np.arange(n_items))
        results = []
        test_history['pred'] = 0
        for user_id in test_history['user_id']:
            print(user_id)
            user_x = user_dict[user_id]
            print(user_x)
            recom = mf_model.predict(user_x, np.arange(n_items), num_threads=4)
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
        interactions = create_interaction_matrix(df=grouped_train_test,user_col='merchant_id',item_col='card_id',rating_col='target_y')
        train_unique=train.drop_duplicates(subset=['card_id'])
        test_unique=test.drop_duplicates(subset=['card_id'])
        item_features= train_unique.append(test_unique)[['feature_1', 'feature_2', 'feature_3',
           'num_transactions', 'sum_trans', 'mean_trans', 'std_trans', 'min_trans',
           'max_trans', 'year_first', 'month_first']]
        mf_model = runMF(interactions=interactions,
                                      n_components=30, loss='warp', epoch=40, n_jobs=4)
        # Create User Dict
        user_dict = create_user_dict(interactions=interactions)

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
            return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x))
        else:
            return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x)), scores_rmse[['card_id','pred']]
    
    
    
    print('FINISH LIGHTFM')
