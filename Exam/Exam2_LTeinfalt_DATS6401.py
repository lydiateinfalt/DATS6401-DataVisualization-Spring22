#Lydia Teinfalt
#4/20/2022
#DATS 6401
#Exam 2

import dash
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math as m
import seaborn as sns

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# #Main
# my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# plt.style.use('whitegrid')

data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
print(data.describe())
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA


#
# a.	Replace all ‘NAN’ entries (if exists) with the average of the column [if numerical] and remove the entire row of observation if the missing data is due to categorical data.
# b.	Replace all missing data entry with the average of the column [if numerical] and remove the entire row of observation if the missing data is due to categorical data.
# c.	Remove the duplicate data entry based on the “date_time” column only.
# d.	Write a python code that shows there is no missing, nan or duplicate entries inside the cleaned dataset.

df = data.copy()
numeric_cols = df.select_dtypes(include='number').columns.to_list()
print(f'Numeric columns in the metro interstate dataframe {numeric_cols}')

for num in numeric_cols:
      df[num].fillna(df[num].mean(), inplace=True)
print(df.describe())
print(df.head())

df1 = df.copy()
df1['date_time'].drop_duplicates(inplace=True)
def clean_data(df):
    dfnull = df.iloc[1:-1].isnull().values.any()
    if dfnull:
        print("Cleaning up data")
        df1 = df.dropna(how='any')
    else:
        print(f"There are no NAN values")
    return df


data = clean_data(df1)

cols = data.columns


#rain, snow, clouds & temp
df3 = data[['rain_1h','snow_1h','clouds_all', 'temp' ]]
features = df3.columns
#=========================
# PCA Analysis
#=========================
Y = data['traffic_volume']
X = df3[features].values
X = StandardScaler().fit_transform(X)
#pca = PCA(n_components = 'mle', svd_solver='full')
pca = PCA(n_components = 4, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
print("Original Dim", X.shape)
print("Transformed Dim", X_PCA.shape)
print(f"explained variance ratio {pca.explained_variance_ratio_}")
plt.figure()
x = np.arange(1,len(pca.explained_variance_ratio_)+1,1)
plt.xticks(x)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.xlabel("Number of Components")
plt.title("PCA Analysis")
plt.show()
print('*'*50)


#==========================================================
# SVD Analysis and Conditional Number on the original data
#==========================================================

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'Original Data: Singular Values {d}')
print(f'Original Data: condition number {LA.cond(X)}')
print('*'*50)


#==========================================================
# SVD Analysis and Conditional Number on the transformed data
#==========================================================

H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'Transformed Data: Singular Values {d_PCA}')
print(f'Transformed Data: condition number {LA.cond(X_PCA)}')
print('*'*50)

#==========================================================
# Construction of reduced dimension dataset
#==========================================================
a,b = X_PCA.shape
column = []
for i in range(b):
    column.append(f'Principal Col {i+1}')
df_PCA = pd.DataFrame(data = X_PCA, columns = column)
df_PCA = pd.concat([df_PCA, Y], axis=1)
print(df_PCA)
print("")

# 4.	(Python) Outlier Detection: [6pts]
# a.	Calculate the Q1 and Q3 for the traffic volume and display the result in your solution manual.
# b.	Calculate the IQR for the traffic volume and display the following message in your solution manual:
# i.	Any traffic volume value more than  or less than is an outlier.
# c.	Does the traffic volume data contain outlier? Use an IQR method to prove your answer. If an outlier exists, it must be removed from the dataset and the rest of the questions in the test must be answered based on the removed [outliers] dataset. Plot the boxplot to verify your answer in the section a & b.

t = pd.DataFrame(data['traffic_volume'])
median = np.median(t['traffic_volume'])
print(median)
lower = t[t['traffic_volume'] < median]

print(lower)

upper = t[t['traffic_volume'] > median]
print(upper)
Q1 = np.median(lower)
Q3 = np.median(upper)
print(f'Q1 for traffic_volume = {Q1}')
print(f'Q3 for traffic_volume = {Q3}')
print("")
IQR = Q3 -Q1
print(f'IQR = {IQR}')
low_outliers = Q1 - 1.5*IQR
high_outliers = Q3 + 1.5*IQR
print(f'Any traffic volume value more than {low_outliers} or less {high_outliers} than is an outlier.')
plt.boxplot(t)
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot for Traffic Volume')
plt.show()



t1 = pd.DataFrame(data['temp'])
median = np.median(t1['temp'])
print(median)
lower = t1[t1['temp'] < median]

print(lower)

upper = t1[t1['temp'] > median]
print(upper)
Q1t = np.median(lower)
Q3t = np.median(upper)
print(f'Q1 for temp = {Q1t}')
print(f'Q3 for temp = {Q3t}')
print("")
IQRt = Q3t -Q1t
print(f'IQR = {IQRt:.2f}')
low_outlierst = Q1t - 1.5*IQRt
high_outlierst = Q3t + 1.5*IQRt
print(f'Any temp more than {low_outlierst:.2f} or less {high_outlierst} than is an outlier.')
plt.boxplot(t1)
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot for Temp')
plt.show()


sns.heatmap(data.corr())
plt.title("Heat Map")
plt.tight_layout()
plt.show()


plt.bar(data['weather_main'],data['traffic_volume'] )
plt.xlabel('Weather')
plt.legend()
plt.ylabel('Traffic Volume')
plt.title("Stack Bar Plot")
plt.grid()
plt.xticks(rotation=90)
plt.show()


sns.histplot(data, x = "traffic_volume", hue='weather_main', element = "poly", bins = 15)
plt.title('Histogram of Weather Types')
plt.show()

sns.histplot(data,  x = "traffic_volume", hue='weather_main',  element = "bars", bins = 15)
plt.title('Histogram of Weather Types')
plt.show()

sns.histplot(data,  x = "traffic_volume", hue='weather_main',  element = "step", bins = 15)
plt.title('Histogram of Weather Types')
plt.show()

sns.histplot(data,  x = "traffic_volume", hue='weather_main',  multiple = "stack", bins = 15)
plt.title('Histogram of Weather Types')
plt.show()

sns.kdeplot(data = data, y="traffic_volume", hue = 'weather_main', fill = True)
plt.show()



