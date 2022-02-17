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

df = pandas.DataFrame(data)

for i in cols:
    count = df[i].isnull().sum()
    ratio = count/len(df)
    if (ratio > .20):
        print(f'Column {i} has greater than 20% NAs')

df = pandas.DataFrame(data)
df.dropna(how='any', inplace = True)
cols = df.columns
print(f'The cleaned dataset has {len(df)} of observations')
print(f'The cleaned dataset has {len(cols)} features [columns]')
print("The list of removed columns are none.")



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
print(f'The mean of tip {total_tip_mean: .2f}')
print(f'The variance of total is {total_bill1: .2f}')
print(f'The variance of tip {tip_var: .2f}')
#print(f'The median of total is {total_bill_median: .2f}')
#print(f'The median of tip {total_tip_median:.2f}')
#print(f'The std of total_bill {total_bill_std: .2f}')
#print(f'The std of tip {tip_std :.2f}')



df['tip_percentage']= df['tip']/df['total']
total_passengers = len(df)
print(f"{len(df[df['tip_percentage'] == 0])/total_passengers} of passengers did not tip at all")
print(f"{len(df[(df['tip_percentage'] >= .10)&(df['tip_percentage']<.15)])/total_passengers} of passengers tipped 10-15% of total [10 is included and 15 is excluded]")
print(f"{len(df[(df['tip_percentage'] >= .15)&(df['tip_percentage']<.20)])/total_passengers} of passengers tipped 15-20% of total [15 is included and 20 is excluded]")
print(f"{len(df[df['tip_percentage'] >= .20])/total_passengers} passengers tipped more than 20% of total [20 is included]")
#majority of passengers tipped "median"
tip_perc_median = df['tip_percentage'].median()
print(f"Majority of passengers tipped {(tip_perc_median)*100:.2f}%")

plt.figure(figsize=(12,8))
plt.hist(df['tip'], bins = 50, label='tip')
plt.hist(df['total'], bins= 50, label='total')
plt.title('Histogram Plot')
plt.grid()
plt.legend()
plt.show()

def corr_coeff(x, y):
    diff_x = []
    diff_y = []
    r = []
    diff_x_squared = []
    diff_y_squared = []

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    for i in range(len(x)):
        x_hat = (x[i] - mean_x)
        diff_x.append(x_hat)
        diff_x_squared.append(np.square(x_hat))
        y_hat = (y[i]- mean_y)
        r.append((x_hat * y_hat))
        diff_y_squared.append(np.square(y_hat))
    return (np.sum(r))/(np.sqrt(np.sum(diff_x_squared))*np.sqrt(np.sum(diff_y_squared)))

tip = df['tip'].values
total = df['total'].values
fare = df['fare'].values
distance = df['distance'].values
r_x_y = corr_coeff(tip, distance)
r_x_y1 = corr_coeff(fare,distance)
r_x_y2 = corr_coeff(tip, fare)
var2 = df.corr()
df_coref = df[["fare", "distance", "tip"]]
print(df_coref.corr())
print(f'The correlation coefficient between fare & distance is {r_x_y1:.2f}')
print(f'The correlation coefficient between tip & fare is {r_x_y2:.2f}')
print(f'The correlation coefficient between tip & distance is {r_x_y:.2f}')
print("The fare amount has the highest correlation coefficient with ")



a = df2.isnull().sum()/len(df2)
print(a)
col = df.columns
variable = []
df1 = df.copy()
for i in range(0, len(col)):
    if a[i] >= .2:
        variable.append((col[i]))

df1.drop(variable, axis = 1, inplace=True)
a1 = df1.isnull().sum()/len(df2)

print(f'The mean of the total is ${df1.total.mean():2f}')
print(f'The mean of the tip is ${df1.tip.mean():2f}')
print(f'The var of the total is ${df1.total.var():2f}')
print(f'The var of the total is ${df1.tip.var():2f}')

df1['tip_perc']=df1.tip/df1.total*100
tip_zero = df1[df1['tip_perc'].values == 0]
tip_20 = df1[df['tip_perc'].values >= 50]
print(f'{len(tip_zero)/len(df1):2f}% of passengers did not tip at all')

df4 = df[["fare", "distance", "tip"]]
corr = df4.corr()
print(corr)





