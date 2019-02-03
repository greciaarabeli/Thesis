import numpy as np
import pandas as pd
import random

order_products_prior = pd.read_csv("order_products__prior.csv")
orders = pd.read_csv("orders.csv")
orders=orders.loc[orders['eval_set']=='prior']
products= pd.read_csv("products.csv")

data=pd.merge(order_products_prior, orders, on='order_id', how='left')
data=pd.merge(data, products, on='product_id', how='left')
users_list=orders.user_id.unique()

random.Random(4).shuffle(users_list)

users_batch={}
n_users_batch=2000
x=0
y=len(users_list)
for i in range(x,y,n_users_batch):
    x=i
    data["batch"] = np.where(data["user_id"].isin(users_list[x:x + n_users_batch]), 'users_%s' % i, data["batch"])

idx_test = data.groupby(['user_id'])['order_number'].transform(max) == data['order_number']
test=data[idx_test]
orders_set_testlist=test.order_id.unique()
train = data[-data["order_id"].isin(orders_set_testlist)]
orders_set_trainlist=train.order_id.unique()

total=train.groupby(['order_id']).size().reset_index(name='total')
train=train.merge(total,on='order_id')

train.to_csv('1_train.csv')
test.to_csv('1_test.csv')
