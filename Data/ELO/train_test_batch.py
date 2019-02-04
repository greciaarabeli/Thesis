import pandas as pd
import numpy as np
import random

new_transactions = pd.read_csv('new_merchant_transactions.csv', parse_dates=['purchase_date'])
merchants = pd.read_csv('merchants.csv')
historical_transactions = pd.read_csv('historical_transactions.csv',parse_dates=['purchase_date'])
test = pd.read_csv('test.csv', parse_dates=["first_active_month"])
train = pd.read_csv('train.csv', parse_dates=["first_active_month"])

historical_transactions=historical_transactions.append(new_transactions)

historical_transactions["batch"] =0
train["batch"] =0
test["batch"] =0

card_list=historical_transactions.card_id.unique()
random.Random(4).shuffle(card_list)

card_batch={}
n_card=2000
x=0
y=len(card_list)
for i in range(x,y,n_card):
    x=i
    print(i)
    historical_transactions["batch"] = np.where(historical_transactions["card_id"].isin(card_list[x:x + n_card]), 'users_%s' % i, historical_transactions["batch"])
    test['batch']=np.where(test["card_id"].isin(card_list[x:x + n_card]), 'users_%s' % i, test["batch"])
    train['batch']=np.where(train["card_id"].isin(card_list[x:x + n_card]), 'users_%s' % i, train["batch"])
    
 train.to_csv('1_train.csv')
 test.to_csv('1_test.csv')
 historical_transactions.to_csv('1_historical_transactions.csv')
 
