""" Function to calculate the next best action according to each data set. 
    For Instacart it gives the  products from last order and for elo it gives the average index from 
    the merchants visited by the user
    Parameters:train
               test
               data
               return_pred: 0 if you just need the accuracy score and 1 if you need the prediction
               dataset: 'instacart' or 'elo'
               
   Return: if 0 return average f1 for instacart and average MSE for elo
           if 1 return the same as 0 plus the predictions for test
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def calculate_prf(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0,0,0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    f1= 2 * p * r / (p + r)
    return p,r,f1


def easiest(train, test, data, return_pred, dataset):
    if dataset=='instacart':

        last_orders = train.groupby("user_id")["order_number"].aggregate(np.max)
        t = pd.merge(left=last_orders.reset_index(), right=train, how='inner', on=['user_id', 'order_number'])
        t_last_order = t.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        t_last_order = pd.merge(t_last_order, train[['order_id', 'user_id']], on='order_id')
        t_last_order = t_last_order.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

        test_history = test[test['reordered']==1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        set1 = set(test.order_id.unique())
        set2 = set(test_history.index)
        missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
        missing['product_id'] = 'NaN'
        missing = missing.rename(index=str, columns={0: "order_id"})
        test_history = test_history.reset_index()
        test_history = test_history.append(missing)
        test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
        test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

        t_last_order = pd.merge(t_last_order, test_history, on='user_id')
        t_last_order = t_last_order.sort_values('user_id').fillna('NaN')
        y_pred=t_last_order['product_id_x']
        y_true=t_last_order['product_id_y']

        sum_recall = 0
        sum_precision = 0
        sum_fscore = 0
        n = len(y_true)
        print(n)
        for i in y_true.index:
            p, r, f1 = calculate_prf(y_true[i], y_pred[i])
            sum_precision = sum_precision + p
            sum_recall = sum_recall + r
            sum_fscore = sum_fscore + f1

        if return_pred==0:
            return sum_fscore/n
        else:
            return sum_fscore/n, y_pred,
    
    
    else:
        train=train.merge(data, on='card_id', how='left')
        train_avgtarget = train.groupby(["merchant_id"])["target"].aggregate("mean").reset_index()
        train=train.merge(train_avgtarget, on='merchant_id', how='left')
        test=test.merge(data, on='card_id', how='left')
        test=test.merge(train_avgtarget, on='merchant_id', how='left').fillna(train_avgtarget.target.mean())
        pred=test.groupby(["card_id"])["target_y"].aggregate("mean").reset_index()
        pred=pred.merge(test[['card_id','target_x']], on='card_id',how='left').drop_duplicates()
        if return_pred==0:
            return np.sqrt(mean_squared_error(pred.target_y, pred.target_x))
        else:
            return np.sqrt(mean_squared_error(pred.target_y, pred.target_x)), pred[['card_id', 'target_y']]
        
    print('FINISH LAST ORDER')
