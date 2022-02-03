import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('seaborn-deep')

# #Start Date
# start_dt = '2000-01-01'
#
# #End Date
# end_dt = '2022-02-02'
#
# #Source to read in stocks data
# source = 'yahoo'
# df = web.DataReader('MSFT', data_source = source, start= start_dt, end = end_dt)
# col = df.columns
# correlation = df.corr()
# print(correlation)
# plt.hexbin(df['Volume'].values, df['Open'].values, gridsize=(50,50))
# plt.show()
np.random.seed(123)
x = np.random.normal(size=5000)
y = np.random.normal(size=5000)
# y = 2*x+ np.random.normal(size=5000)
plt.figure()
plt.hexbin(x,y, gridsize=(50,50))
plt.xlabel('Random Variable x', fontsize = 20)
plt.ylabel('Random Variable y', fontsize = 20)
plt.title("Hexbin plot between Normal random Variables", fontsize = 25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('Test.pdf', dpi = 600)
plt.show()
