import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('seaborn-deep')

#Start Date
start_dt = '2000-01-01'

#End Date
end_dt = '2022-02-02'

#Source to read in stocks data
source = 'yahoo'
df = web.DataReader('MSFT', data_source = source, start= start_dt, end = end_dt)
col = df.columns
print(col)
col2 = col.drop('Volume')

df[col2].plot()
plt.show()