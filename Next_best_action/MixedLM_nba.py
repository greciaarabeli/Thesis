""" Function to calculate the next best action according to each data set. 
    For Instacart it uses Classifier and for Elo uses Regression
    Parameters:train
               test
               data
               return_pred: 0 if you just need the accuracy score and 1 if you need the prediction
               dataset: 'instacart' or 'elo'
               
   Return: if 0 return average f1 for instacart and average MSE for elo
           if 1 return the same as 0 plus the predictions for test
"""

from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd

def calculate_prf(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0,0,0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    f1= 2 * p * r / (p + r)
    return p,r,f1

def do_MixedLM(train, test,data, return_pred, dataset): 

    
    ### INSTACART ###
    if dataset=='instacart':
        orders_set_test=test.order_id.unique()
        
        cols_to_use=['user_id','order_id', 'product_id', 'order_number', 'order_dow', 'order_hour_of_day',
       'days_since_prior_order', 'aisle_id', 'department_id',
        'Totalmin', 'Totalmax', 'Totalmean',
       'order_numbermax', 'days_since_prior_ordermin',
       'days_since_prior_ordermax', 'days_since_prior_ordermean',
       'reordermin', 'reordermax', 'reordermean',
       'order_hour_of_daymin', 'order_hour_of_daymax', 'order_hour_of_daymean',
       'order_dowmean']
        
        target_col=['reordered']
        
        y_train = train[target_col]
        X_train = train.drop(['reordered', 'eval_set', 'total','product_name', 'add_to_cart_order'], axis=1)

        X_test = test.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
        X_test=X_test.drop(['product_id','add_to_cart_order', 'reordered', 'eval_set', 'product_name', 'aisle_id', 'department_id'], axis=1)

        X_train_sub=train.drop_duplicates(subset=['product_id', 'user_id'], keep='first')
        X_train_sub=X_train_sub[['product_id', 'user_id', 'aisle_id','department_id', 'Totalmin', 'Totalmax', 'Totalmean',
               'order_numbermax', 'days_since_prior_ordermin',
               'days_since_prior_ordermax', 'days_since_prior_ordermean','reordermin', 'reordermax', 'reordermean',
               'order_hour_of_daymin', 'order_hour_of_daymax', 'order_hour_of_daymean',
               'order_dowmean']]

        X_test=pd.merge(left=X_test, right=X_train_sub, how='right',on=['user_id'])
        X_test = X_test[cols_to_use]

        X_train = X_train[cols_to_use]

        if 'cluster' in train.columns:
            ## Mixture Linear Model
            model =MixedLM(endog=y_train, exog=X_train, groups=train['cluster'])
            result = model.fit()

            predict_test = pd.DataFrame({"user_id":test["user_id"].values})
            predict_test["target"] = pd.DataFrame(model.predict(result.fe_params, exog=X_test))

            predict_test['pred'] = np.where(predict_test['target']>np.mean(predict_test.target), 1, 0)

            X_test['pred']=predict_test.pred
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

                p, r, f1 = calculate_prf(y, y_hat)
                sum_precision = sum_precision + p
                sum_recall = sum_recall + r
                sum_fscore = sum_fscore + f1


            if return_pred==0:
                return sum_fscore/n
            else:
                return sum_fscore/n, y_pred
            
        else:
            ## Linear Model
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            predict_test = pd.DataFrame({"user_id":test["user_id"].values})
            predict_test["target"] = pd.DataFrame(regr.predict(X_test))

            predict_test['pred'] = np.where(predict_test['target']>np.mean(predict_test.target), 1, 0)

            X_test['pred']=predict_test.pred
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

                p, r, f1 = calculate_prf(y, y_hat)
                sum_precision = sum_precision + p
                sum_recall = sum_recall + r
                sum_fscore = sum_fscore + f1


            if return_pred==0:
                return sum_fscore/n
            else:
                return sum_fscore/n, y_pred
          

       

    else:
        ### ELO ###
        cols_to_use=['feature_1', 'feature_2', 'feature_3',
        'num_transactions', 'sum_trans', 'mean_trans',
       'std_trans', 'min_trans', 'max_trans', 'year_first', 'month_first']
        target_col=['target']
    
        X_train=train[cols_to_use]
        X_test=test[cols_to_use]
        y_train=train['target']
        y_test=test['target']
        
        if 'cluster' in train.columns:
            ## Mixture Linear Model
            model =MixedLM(endog=y_train, exog=X_train, groups=train['cluster'])
            result = model.fit()

            predict_test = pd.DataFrame({"card_id":test["card_id"].values})
            predict_test["target"] = pd.DataFrame(model.predict(result.fe_params, exog=X_test))
            if return_pred==0:
                return np.sqrt(mean_squared_error(predict_test.target.values, y_test))
            else:
                return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_test
            
        else:
            #Linear Model
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            predict_test = pd.DataFrame({"card_id":test["card_id"].values})
            predict_test["target"] = pd.DataFrame(regr.predict(X_test))
            if return_pred==0:
                return np.sqrt(mean_squared_error(predict_test.target.values, y_test))
            else:
                return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_test
            
            
    print('FINISH MERF')
