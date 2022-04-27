import numpy as np
import plotly.express as px
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
url = "https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/autos.clean.csv"


df = pd.read_csv(url)
print(f"The original shape of the dataset is {df.shape}")
X = df[df._get_numeric_data().columns.to_list()][:-1]
Y = df['price']

features = X.columns

#=========================
# PCA Analysis
#=========================
X = X[features].values
X = StandardScaler().fit_transform(X)
#pca = PCA(n_components = 'mle', svd_solver='full')
pca = PCA(n_components = 7, svd_solver='full')
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



