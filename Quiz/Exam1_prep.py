import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web

today = datetime.date.today()
stocks = ['TSLA', 'AMZN','DIS']
df1 = pd.DataFrame()
for i in stocks:
    df = web.DataReader(i, data_source = 'yahoo', start='01-01-2019', end=today)
    df['Symbol'] = i
    df1 = df1.append(df)

renamed_cols = {'High': 'High($)', 'Low': 'Low($)', 'Open': 'Open($)', 'Close': 'Close($)', 'Adj Close' : 'Adj Close($)'}
columns = df1.columns
col_list = columns.tolist()
col_list.remove('Symbol')
df_mean = df1.groupby('Symbol').mean()
df_mean= pd.DataFrame(df_mean, index=['TSLA', 'AMZN','DIS'])
df_mean.rename(columns=renamed_cols, inplace=True)
print(df_mean)

df_copy = df_mean.copy()
df_copy['Max'] = df_mean.idxmax(axis=1)
df_copy['Min'] = df_mean.idxmin(axis=1)
df_copy = df_copy.round(decimals=2)
print(df_copy)

df_std = df1.groupby('Symbol').std()
df_std= pd.DataFrame(df_std, index=['TSLA', 'AMZN','DIS'])
df_std.rename(columns=renamed_cols, inplace=True)

df_copy1 = df_std.copy()
df_copy['Max'] = df_std.idxmax(axis=1)
df_copy['Min'] = df_std.idxmin(axis=1)
df_copy1 = df_copy1.round(decimals=2)
print(df_copy1)

df2 = sns.load_dataset("penguins")
print(df2)
total_na = df2.isnull().sum()
print(total_na)

col_list = df2.columns
print(col_list)
for i in col_list:
    null_values = df2[i].isnull().sum()
    ratio = null_values/len(df2)
    if (ratio > 0.02):
        popular = df2[i].mode()
        print(popular)
        df2[i].fillna(popular, inplace=True)

print(df2)

