""" Function to calculate the next best action according to each data set using xgboost. 
    For Instacart it uses Binary logistic and for Elo uses Linear Regression
    Parameters:train
               test
               data
               return_pred: 0 if you just need the accuracy score and 1 if you need the prediction
               dataset: 'instacart' or 'elo'
               
   Return: if 0 return average f1 for instacart and average MSE for elo
           if 1 return the same as 0 plus the predictions for test


import xgboost
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def do_xgboost(train, test, data, return_pred, dataset):
    
    if dataset=='instacart':
        param = {'max_depth': 10,
                 'eta': 0.02,
                 'colsample_bytree': 0.4,
                 'subsample': 0.75,
                 'silent': 1,
                 'nthread': 27,
                 'eval_metric': 'logloss',
                 'objective': 'binary:logistic',
                 'tree_method': 'hist'
                 }
        orders_set_test=test.order_id.unique()
        y_train = train['reordered']
        X_train = train.drop(['reordered', 'eval_set', 'total','product_name', 'add_to_cart_order'], axis=1)

        X_test = test.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
        X_test=X_test.drop(['product_id','add_to_cart_order', 'reordered', 'eval_set', 'product_name', 'aisle_id', 'department_id'], axis=1)
        X_train_sub=train.drop_duplicates(subset=['product_id', 'user_id'], keep='first')
        X_train_sub=X_train_sub[['product_id', 'user_id', 'aisle_id','department_id']]
        X_test=pd.merge(left=X_test, right=X_train_sub, how='right',on=['user_id'])
        X_test = X_test[['order_id', 'product_id', 'user_id', 'order_number', 'order_dow','order_hour_of_day', 'days_since_prior_order', 'aisle_id','department_id']]
        X_train = X_train[['order_id', 'product_id', 'user_id', 'order_number', 'order_dow','order_hour_of_day', 'days_since_prior_order', 'aisle_id','department_id']]
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test)
        model = xgboost.train(param, dtrain)
        predict_labels = model.predict(dtest)

        X_test['pred']=predict_labels
        X_test['pred'] = (X_test['pred'] > X_test['pred'].mean()) * 1
        pred = X_test[X_test['pred'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        true =test[test['reordered'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
        sum_recall = 0
        sum_precision = 0
        sum_fscore = 0
        n = len(orders_set_test)
        for i in orders_set_test:
            if i in true.index:
                y = true['product_id'][i]
            else:
                y = 'nan'
            if i in pred.index:
                y_hat = pred['product_id'][i][:10]
            else:
                y_hat = 'nan'

            p, r, f1 = Evrecsys.calculate_prf(y, y_hat)
            sum_precision = sum_precision + p
            sum_recall = sum_recall + r
            sum_fscore = sum_fscore + f1
            
        if return_pred==0:
            return sum_recall/n,sum_precision/n ,sum_fscore/n
        else:
            return sum_recall/n,sum_precision/n ,sum_fscore/n, y_pred


    else:
        param = {'max_depth': 10,
                     'eta': 0.02,
                     'colsample_bytree': 0.4,
                     'subsample': 0.75,
                     'silent': 1,
                     'nthread': 27,
                     'eval_metric': 'rmse',
                     'objective': 'reg:linear',
                     'tree_method': 'hist'
                     }
        cols_to_use=['feature_1', 'feature_2', 'feature_3',
        'num_transactions', 'sum_trans', 'mean_trans',
       'std_trans', 'min_trans', 'max_trans', 'year_first', 'month_first']
        target_col=['target']

        X_train=train[cols_to_use]
        X_test=test[cols_to_use]
        y_train=train[target_col].values
        y_test=test[target_col].values

        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test)
        model = xgboost.train(param, dtrain)
        predict_test = pd.DataFrame({"card_id":test["card_id"].values})
        predict_test["target"] = pd.DataFrame(model.predict(dtest))
        if return_pred==0:
            return np.sqrt(mean_squared_error(predict_test.target.values, y_test))
        else:
            return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_test
    print('FINISH XGBOOST')
