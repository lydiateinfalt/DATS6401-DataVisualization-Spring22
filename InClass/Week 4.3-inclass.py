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

fig =plt.figure(figsize=(16,8))
for i in range(1,7):
    ax1 = fig.add_subplot(2,3,i)
    ax1.plot(df[col[i-1]].values)
    ax1.set_xlabel('Time', fontsize = 15)
    if col[i-1] =='Volume':
        ax1.set_ylabel('Quantity')
    else: ax1.set_ylabel('USD ($)')
    ax1.set_title(f'MSFT {col[i-1]}', fontsize=15)
    ax1.set_font=20
    plt.grid(visible=True, axis = 'y')
plt.tight_layout()
plt.show()