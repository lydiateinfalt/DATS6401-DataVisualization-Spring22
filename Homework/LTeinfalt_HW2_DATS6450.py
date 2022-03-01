# DATS 6401: Homework 1 (Spring 22)
# Lydia Teinfalt
# 02/03/2022
import datetime

import matplotlib.pyplot
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from pandas.plotting import scatter_matrix

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

#Start Date
start_dt = '2000-01-01'

#End Date
end_dt = datetime.datetime.today()

#Source to read in stocks data
source = 'yahoo'

#Create empty dataframe
df1 = pd.DataFrame()

for i in range(len(stocks)):
     df = web.DataReader(stocks[i], data_source = source, start= start_dt, end = end_dt)
     df['Symbol'] = stocks[i]
     df1 = df1.append(df)
#print(df1)
col = df1.columns
print(df1.head())

#Line Plots
# for j in col[0:6]:
#       df_stock = df1[[j, 'Symbol']]
#       fig = plt.figure(figsize=(16, 8))
#       i = 1
#       for s in stocks:
#           ax1 = fig.add_subplot(2,3,i)
#           data = df_stock[df_stock['Symbol'] == s]
#           ax1.plot(data.index.values, data[j].values)
#           ax1.set_xlabel('Date', fontsize=15)
#           if j == 'Volume':
#               ax1.set_ylabel('Quantity')
#               ax1.set_title(f'{j} history of {s} ', fontsize=15)
#           else:
#               ax1.set_ylabel(f'{j} price USD($) ')
#               ax1.set_title(f'{j} price history of {s} ', fontsize=15)
#           ax1.set_font = 20
#           plt.grid(visible=True)
#           i += 1
#       plt.tight_layout()
#       plt.show()
#
# # #Histograms
# for j in col[0:6]:
#      df_stock = df1[[j, 'Symbol']]
#      fig = plt.figure(figsize=(16, 8))
#      i = 1
#      for s in stocks:
#          ax1 = fig.add_subplot(2,3,i)
#          data = df_stock[df_stock['Symbol'] == s]
#          ax1.hist(data[j].values, bins=50)
#          if j == 'Volume':
#              ax1.set_ylabel('Frequency')
#              ax1.set_title(f'{j} history of {s} ', fontsize=15)
#              ax1.set_xlabel('Quantity', fontsize=15)
#          else:
#              ax1.set_ylabel('Frequency')
#              ax1.set_title(f'{j} price history of {s} ', fontsize=15)
#              ax1.set_xlabel('Value in USD($)', fontsize=15)
#          ax1.set_font = 20
#          plt.grid(visible=True)
#          i += 1
#      plt.tight_layout()
#      plt.show()
#
# for org in stocks:
#     correlation = df1[df1['Symbol'] == org].corr(method='pearson')
#     print(correlation)
#     corr_pairs = correlation.unstack().sort_values(ascending=False).drop_duplicates()
#     print(corr_pairs[2])
#     print(f"{org} {corr_pairs.index[1]} have highest correlation coefficient of {corr_pairs[1]:.2f}")
#     print(f"{org} {corr_pairs.index[-1]} lowest correlation coefficient of {corr_pairs[-1]:.2f}")
#
#
# feature = col.to_list()
# feature.remove('Symbol')
# print(f"Features: {feature}")
# for co in stocks:
#     print(f"Company: {co}")
#     fig = plt.figure(figsize=(16, 16))
#     dfc = df1[df1['Symbol'] == co]
#     index = 1
#     for t in range(0,6):
#         f1 = dfc[feature[t]]
#         for v in range(0,6):
#             ax1 = fig.add_subplot(6, 6, index)
#             f2 = dfc[feature[v]]
#             plt.grid()
#             if (feature[t] == 'Volume'):
#                 plt.xlabel(feature[v])
#                 plt.ylabel(feature[t])
#                 plt.scatter(f2, f1)
#             else:
#                 plt.xlabel(feature[t])
#                 plt.ylabel(feature[v])
#                 plt.scatter(f1, f2)
#             plt.title(f"r={f2.corr(f1):.2f}")
#             index +=1
#     plt.tight_layout()
#     plt.show()
#     print(co)
#
#


df10 = df1[df1['Symbol'] == 'AAPL']
plt.figure(figsize=(16,16))
pd.plotting.scatter_matrix(df10,hist_kwds= {'bins' : 50} , alpha = 0.5, s = 10, diagonal = 'kde')
plt.grid()
plt.tight_layout()
plt.show()

