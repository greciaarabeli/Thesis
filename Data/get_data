import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Read data by chunks
def valid(chunks, batch):
    for chunk in chunks:
        print(chunk.shape)
        mask = chunk['batch'] == batch
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]

# Fill missing data
def missing_impute(df):
    for i in df.columns:
        if df[i].dtype == "object":
            df[i] = df[i].fillna("other")
        elif (df[i].dtype == "int64" or df[i].dtype == "float64"):
            df[i] = df[i].fillna(df[i].mean())
        else:
            pass
    return df



""" Function that prepares the data set that is require
    Parameters:
          dataset_name: 'instacart' or 'elo'
          batch: indicate the batch which one you want to work (ex: 'users_2000')

    Returns:
          train: train dataset read for use it in clustering techniques
          test:  test dataset ready for use it in clustering techniques
          data:  data is just usefull for ELO data set is the detailed information of transaction corresponding to card_id in the batch """




def get_data_batch(dataset_name, batch):
    if dataset_name=='instacart':
        chunksize = 10 ** 6
        chunks_train = pd.read_csv("instacart/1_train.csv", chunksize=chunksize, index_col=0)
        train = pd.concat(valid(chunks_train, batch))

        chunks_test = pd.read_csv("instacart/1_test.csv", chunksize=chunksize, index_col=0)
        test = pd.concat(valid(chunks_test, batch))

        data=0

    else:
        chunksize = 10 ** 6
        chunks_train = pd.read_csv("elo/1_train.csv", parse_dates=["first_active_month"],chunksize=chunksize)
        train = pd.concat(valid(chunks_train, batch))

        chunks_test = pd.read_csv("elo/1_test.csv", parse_dates=["first_active_month"], chunksize=chunksize)
        test = pd.concat(valid(chunks_test, batch))

        data= pd.read_csv("elo/1_historical_transactions.csv",parse_dates=['purchase_date'], chunksize=chunksize)
        historical_trans= pd.concat(valid(data, batch))

        merchants = pd.read_csv('elo/merchants.csv')

        for df in [train, test, merchants]:  # , data]:
            missing_impute(df)

        le = preprocessing.LabelEncoder()
        le.fit(merchants['category_1'])
        merchants['category_1'] = le.transform(merchants['category_1'])

        le.fit(merchants['most_recent_sales_range'])
        merchants['most_recent_sales_range'] = le.transform(merchants['most_recent_sales_range'])

        le.fit(merchants['most_recent_purchases_range'])
        merchants['most_recent_purchases_range'] = le.transform(merchants['most_recent_purchases_range'])

        le.fit(merchants['category_4'])
        merchants['category_4'] = le.transform(merchants['category_4'])

        # number of transactions
        gdf = data.groupby("card_id")
        gdf = gdf["purchase_amount"].size().reset_index()
        gdf.columns = ["card_id", "num_transactions"]
        train = pd.merge(train, gdf, on="card_id", how="left")
        test = pd.merge(test, gdf, on="card_id", how="left")

        # Stadistics about purchase amount in new merch
        gdf = data.groupby("card_id")
        gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
        gdf.columns = ["card_id", "sum_trans", "mean_trans", "std_trans", "min_trans", "max_trans"]
        train = pd.merge(train, gdf, on="card_id", how="left")
        test = pd.merge(test, gdf, on="card_id", how="left")

        train["year_first"] = train["first_active_month"].dt.year
        test["year_first"] = test["first_active_month"].dt.year
        train["month_first"] = train["first_active_month"].dt.month
        test["month_first"] = test["first_active_month"].dt.month
        data["year_purch"] = data["purchase_date"].dt.year
        data["month_purch"] = data["purchase_date"].dt.month
        data["year_month_purch"] = data["purchase_date"].dt.strftime('%Y/%m')

        train, test = train_test_split(train, test_size=0.3)


    return train, test, data



    
