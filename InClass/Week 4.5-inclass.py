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
# plt.figure(figsize=(16,8))
# fig, ax = plt.subplots(2, 3, figsize=(12,8))
# z=0
# for i in range(1,3):
#     for j in range (1,4):
#         ax[i-1,j-1].plot(df[col[z]].values)
#         ax[i-1,j-1].set_title(f"MSFT {col[z]}")
#         ax[i-1,j-1].set_xlabel("Time")
#         ax[i-1,j-1].set_ylabel("USD ($)")
#         ax[i-1,j-1].grid()
#         z += 1
# plt.tight_layout()
# plt.show()

pd.plotting.scatter_matrix(df, diagonal = 'hist',
                    hist_kwds = {'bins': 50})
plt.show()