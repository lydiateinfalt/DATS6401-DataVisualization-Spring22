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

plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
plt.plot(df.Close)
plt.grid(axis ='y')
plt.xlabel('Time')
plt.ylabel('USD ($)')
plt.title('MSFT Close')

plt.subplot(2,3,2)
plt.plot(df.Low)
plt.xlabel('Time')
plt.grid(axis ='y')
plt.ylabel('USD ($)')
plt.title('MSFT Low')


plt.subplot(2,3,3)
plt.plot(df.Volume)
plt.xlabel('Time')
plt.grid(axis ='y')
plt.ylabel('Quantity')
plt.title('MSFT Volume')


plt.subplot(2,3,4)
plt.plot(df.High)
plt.xlabel('Time')
plt.grid(axis ='y')
plt.ylabel('USD ($)')
plt.title('MSFT High')



plt.subplot(2,3,5)
plt.plot(df['Adj Close'].values)
plt.xlabel('Time')
plt.grid(axis ='y')
plt.ylabel('USD ($)')
plt.title('MSFT Adjusted Close')


plt.subplot(2,3,6)
plt.plot(df['Open'])
plt.xlabel('Time')
plt.grid(axis ='y')
plt.ylabel('USD ($)')
plt.title('MSFT Open')
plt.tight_layout()
plt.show()