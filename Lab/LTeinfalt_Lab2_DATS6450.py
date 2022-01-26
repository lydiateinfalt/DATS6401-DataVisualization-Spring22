import pandas_datareader as web
import matplotlib.pyplot as plt

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

for i in range(len(stocks)):
    df = web.DataReader(stocks[i],data_source = 'yahoo', start='2000-01-01', end ='2021-09-18')

print(df)


