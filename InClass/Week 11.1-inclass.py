import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd

#Start Date
start_dt = '2000-01-01'

#End Date
end_dt = '2022-04-13'

#Source to read in stocks data
source = 'yahoo'

#Create empty dataframe
df = web.DataReader('AAPL', data_source = source, start= start_dt, end = end_dt)

df[['Close', 'Volume']].plot()
plt.show()

close = df['Close'].values
volume = df['Volume'].values

close_z = (close - np.mean(close))/np.std(close)
volume_z = (volume - np.mean(volume))/np.std(volume)
plt.figure()
plt.plot(df.index, close_z, label = 'close')
plt.plot(df.index, volume_z, label = 'volume')
plt.legend()
plt.show()

y = np.array([1,3,5,4,7,9])
mean_y = np.mean(y)
std_y = np.std(y)
z = (y - mean_y)/std_y
print(z)
