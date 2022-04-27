# DATS 6401: Lab 4 (Spring 22)
# Lydia Teinfalt
# 03/12/2022

import plotly.express as px
import pandas_datareader as web
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import seaborn as sns

stocks = px.data.stocks()
columns = stocks.columns
print(stocks.head())

# There are six giant tech companies in this dataset. Plot the stock values versus time in one graph. The x- axis is the date and the y axis is the stock value. Update the layout with the following settings:
# a. Font_color = ‘red’
# b. Legend_title_font_color = ‘green’
# c. font_family = ’ Courier New’
# d. title_font_family = ‘Times New Roman’


fig = px.line(stocks, x = 'date', y = columns, title='Stock Values - Major Tech Company')
fig.update_layout(
    font_color="blue",
    legend_title_font_color="green",
    font_family="Courier New",
    title_font_family="Times New Roman",
    title_font_color="blue",
)
fig.show(renderer = 'browser')

#
fig = make_subplots(rows=3, cols = 2)
GOOG = go.Histogram(x=stocks['GOOG'],nbinsx = 50, name='GOOG')
AAPL = go.Histogram(x=stocks['AAPL'],nbinsx = 50, name='AAPL')
AMZN = go.Histogram(x=stocks['AMZN'],nbinsx = 50, name='AMZN')
FB = go.Histogram(x=stocks['FB'],nbinsx = 50, name='FB')
NFLX = go.Histogram(x=stocks['NFLX'],nbinsx = 50, name='NFLX')
MSFT = go.Histogram(x=stocks['MSFT'],nbinsx = 50, name='MSFT')
fig.append_trace(GOOG, 1, 1)
fig.append_trace(AAPL, 1, 2)
fig.append_trace(AMZN, 2, 1)
fig.append_trace(FB, 2, 2)
fig.append_trace(NFLX, 3, 1)
fig.append_trace(MSFT, 3, 2)
fig.show(renderer = 'browser')

# 4. Consider each company stock as a feature that needs to be fed to a ML model. The target is not given in this problem.
# You need to perform a complete PCA analysis of the ‘stocks’ dataset and answer the following questions and tasks:
# a. Using the following library standard (normalize) the feature space from sklearn.preprocessing import StandardScaler
# b. Find the singular values and condition number for the original feature space.
# c. Find the correlation coefficient matrix between all feature of the original feature space and use the seaborn heatmap to display the result.
# The heatmap for this question should be look like bellow:
features = columns.to_list()[1:]
X = stocks[features].values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components = 'mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
print("Original Dim", X.shape)
print("Transformed Dim", X_PCA.shape)
print(f"explained variance ratio {pca.explained_variance_ratio_}")
print('*'*100)

plt.figure()
x = np.arange(1,len(pca.explained_variance_ratio_)+1,1)
plt.xticks(x)
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.xlabel('number of components')
plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
plt.show(renderer = 'browser')

print('*'*100)
#==========================================================
# SVD Analysis and Conditional Number on the original data
#==========================================================

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'Original Data: Singular Values {d}')
print(f'Original Data: condition number {LA.cond(X)}')
print('*'*100)
#==========================================================
# SVD Analysis and Conditional Number on the transformed data
#==========================================================

H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'Transformed Data: Singular Values {d_PCA}')
print(f'Transformed Data: condition number {LA.cond(X_PCA)}')
print('*'*100)


sns.heatmap(stocks[features].corr())
plt.title('Correlation Coefficients between Features - Original Feature Space')
plt.show(renderer = 'browser')

#==========================================================
# Construction of reduced dimension dataset
#==========================================================
a,b = X_PCA.shape
df_reduced = pd.DataFrame(data = X_PCA)
column = []
for i in range(b):
    column.append(f'Principal Col {i+1}')
df_PCA = pd.DataFrame(data = X_PCA, columns = column)


print(df_PCA.head())
#==========================================================
sns.heatmap(df_PCA.corr())
plt.title('Correlation Coefficients between Features - Reduced Feature Space')
plt.tight_layout()
plt.show()
#==========================================================
df_PCA['date'] = stocks['date']
reduced_cols = df_PCA.columns[:5]
fig = px.line(df_PCA, x = 'date', y =reduced_cols , title='Stock Values - Reduced Feature Space')
fig.update_layout(
    font_color="blue",
    legend_title_font_color="green",
    font_family="Courier New",
    title_font_family="Times New Roman",
    title_font_color="blue",
)
fig.show(renderer = 'browser')


fig = make_subplots(rows=4, cols = 1)
GOOG = go.Histogram(x=df_PCA.iloc[0], name='Principal Col 1')
AAPL = go.Histogram(x=df_PCA.iloc[1],name='Principal Col 2')
AMZN = go.Histogram(x=df_PCA.iloc[2],name='Principal Col 3')
FB = go.Histogram(x=df_PCA.iloc[3],name='Principal Col 4')
fig.append_trace(GOOG, 1, 1)
fig.append_trace(AAPL, 2, 1)
fig.append_trace(AMZN, 3, 1)
fig.append_trace(FB, 4, 1)
fig.show(renderer = 'browser')

fig = px.scatter_matrix(stocks[features])
plt.title("Original Feature Space")
plt.tight_layout()
fig.show(renderer = 'browser')

fig = px.scatter_matrix(df_PCA[column])
plt.title("Reduced Feature Space")
plt.tight_layout()
fig.show(renderer = 'browser')