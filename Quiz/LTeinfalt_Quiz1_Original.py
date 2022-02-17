#Lydia Teinfalt
#Quiz 1

import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy as np

data = sns.load_dataset('taxis')
print(data.head())
cols = data.columns
print(f'There are {len(data)} observations inside the dataset')
print(f'There are {len(cols)} features [columns] inside the dataset')

for i in cols:
    columns = np.where(data[i].isna(), 1, 0)
    df = pandas.DataFrame(columns)
    df.columns = ['NA']
    na_count = df[df['NA'] == 1]
    ratio = len(na_count)/len(df)
    if (ratio > .20):
        print(f'Column {i} has greater than 20% NAs')

df = pandas.DataFrame(data)
df.dropna(how='any', inplace = True)
cols = df.columns
print(f'The cleaned dataset has {len(df)} of observations')
print(f'The cleaned dataset has {len(cols)} features [columns]')
print("The list of removed columns are none.")

# Find the mean & variance of the ‘total’ and ‘tip’ and display a message on the console with 2-digit precision. Display the information on the console
# The mean of the total is ____ $
# The mean of the tip is ______$
# The variance of the total is ____$
# The variance of the tip is _______$

#mean
total_meal_mean = np.mean(df['total'].values)
total_tip_mean = df['tip'].mean()

#variance
total_bill_var = df['total'].var()
total_bill1 = np.var(df['total'])
tip_var = df['tip'].var()

#median
total_bill_median = np.median(df['total'])
total_tip_median = np.median(df['tip'])

#std
total_bill_std = np.std(df['total'])
tip_std = df['tip'].std()

print(f'The mean of total is {total_meal_mean: .2f}')
print(f'The variance of total is {total_bill1: .2f}')
print(f'The median of total is {total_bill_median: .2f}')

print(f'The mean of tip {total_tip_mean: .2f}')
print(f'The variance of tip {tip_var: .2f}')
print(f'The median of tip {total_tip_median:.2f}')

print(f'The std of total_bill {total_bill_std: .2f}')
print(f'The std of tip {tip_std :.2f}')


df['tip_percentage']= np.where((df['tip'] == 0.00 ), 'No Tip','1')





