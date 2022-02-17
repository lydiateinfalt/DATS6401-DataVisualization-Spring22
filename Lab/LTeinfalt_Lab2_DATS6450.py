# DATS 6401: Lab 2 (Spring 22)
# Lydia Teinfalt
# 02/02/2022
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Stock symbols for lab
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

#Start Date
start_dt = '2000-01-01'

#End Date
end_dt = '2021-09-18'

#Source to read in stocks data
source = 'yahoo'

#Create empty dataframe
df1 = pd.DataFrame()

for i in range(len(stocks)):
    df = web.DataReader(stocks[i], data_source = source, start= start_dt, end = end_dt)
    df['Symbol'] = stocks[i]
    df1 = df1.append(df)

columns = ['High','Low', 'Open', 'Close', 'Volume', 'Adj Close']
renamed_cols = {'High': 'High($)', 'Low': 'Low($)', 'Open': 'Open($)', 'Close': 'Close($)', 'Adj Close' : 'Adj Close($)'}
def max_min(dataframe, filename):
    max = []
    min= []

    for i in columns:
        max.append(dataframe[i].max())
        min.append(dataframe[i].min())

    max_series = pd.Series(max, index=dataframe.columns)
    min_series = pd.Series(min, index=dataframe.columns)
    dataframe = dataframe.append(max_series, ignore_index=True)
    dataframe = dataframe.append(min_series, ignore_index=True)
    dataframe.rename(columns=renamed_cols, inplace=True)
    dataframe = dataframe.round(decimals=2)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    dataframe.insert(loc=0, column='Name', value=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT', 'Maximum Value', 'Minimum Value'])
    dataframe.to_csv(filename)
    print(dataframe)

def max_min_company(df, fn):
    df3 = df.copy()
    df3 = df3.round(decimals=2)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    max_vals = df.idxmax(axis=0)
    min_vals = df.idxmin(axis=0)
    df3 = df3.append(max_vals, ignore_index=True)
    df3 = df3.append(min_vals, ignore_index=True)
    df3.insert(loc=0, column='Name',value=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT', 'Company Max Value', 'Company Min Value'])
    df3.rename(columns=renamed_cols, inplace=True)
    df3.to_csv(fn)
    print(df3)

print("*"*64)
# Create Mean Value Comparison Table
print("Mean Value Comparison")
df_mean = df1.groupby('Symbol').mean()
df_mean= pd.DataFrame(df_mean, index=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT'])



max_min(df_mean, "Stocks_Mean.csv")
max_min_company(df_mean, "Company_Mean.csv")

print("*"*64)

# Create Mean Value Comparison Table
print("Standard Deviation Value Comparison")
df_var = df1.groupby('Symbol').var()
df_var= pd.DataFrame(df_var, index=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT'])

max_min(df_var, "Stocks_Cov.csv")
max_min_company(df_var, "Company_variance.csv")
print("*"*64)

#Standard Deviation Value Comparison Table
print("Variance Value Comparison")
df_var = df1.groupby('Symbol').std()
df_var= pd.DataFrame(df_var, index=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT'])

max_min(df_var, "Stocks_STD.csv")
max_min_company(df_var, "Company_STD.csv")
print("*"*64)
#Standard Deviation Value Comparison Table
print("Variance Value Comparison")
df_med = df1.groupby('Symbol').median()
df_med= pd.DataFrame(df_med, index=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT'])

max_min(df_med, "Stocks_Median.csv")
max_min_company(df_med, "Company_Median.csv")
print("*"*64)

Stock_Co = {"AAPL": "Apple", "ORCL": "Oracle", "TSLA": "Tesla", "IBM": "IBM", "YELP": "Yelp", "MSFT": "Microsoft"}

for j in stocks:
    df_corr = df1[df1['Symbol'] == j]
    print("Correlation Matrix for " + Stock_Co[j])
    #dc = df_corr.corr().round(decimals=2)
    dc = df_corr.corr()
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    dc.rename(columns=renamed_cols, inplace=True)
    dc.to_csv(j + ".csv")
    print(dc)
