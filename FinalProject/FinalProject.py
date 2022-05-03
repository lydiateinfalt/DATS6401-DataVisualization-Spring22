import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
from scipy.stats import normaltest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.chdir("C:\\Users\\Teinfalt\\Documents\\GitHub\\DATS6401-DataVisualization-Spring22\\FinalProject")

#--------------Read in data ------------------#
data = pd.read_csv("secondary_data_generated.csv", sep=';', header=0)
print(data.info)
print(data.describe())
columns = data.columns
print(columns)
df = data.copy()

#--------------Preprocess data------------------------------------------#
#Convert stem-width column to from mm to cm in keep with stem-height (cm)
df['stem-width'] = df['stem-width']/10

numeric_cols = df.select_dtypes(include='number').columns.to_list()
categorical_cols = df.select_dtypes(exclude='number').columns.to_list()
print(f'Numeric columns in the mushroom dataframe {numeric_cols}')
print(f'Categorical columns in the mushroom dataframe {numeric_cols}')

#replace null values for numerical columns with the mean
for num in numeric_cols:
    df[num].fillna(df[num].mean(), inplace=True)

a = df.isnull().sum() / len(df)
variable = []

#if the column contain more than 20% null values, drop the columns
for i in range(0, len(columns)):
    if a[i] >= .2:
        variable.append(columns[i])
df.drop(variable, axis=1, inplace=True)

#drop rows where ring-type column have null values (2471)
df.dropna(subset = ['ring-type'], inplace=True)

#replace nan values for gill-attachment with the letter 'u' = unknown
df['gill-attachment'].fillna(value='u', inplace=True)
dfnull=df[df.isna().any(axis=1)]
null_lenth = len(dfnull)
print(f'Number of NULL values in the dataset = {null_lenth}')
print(df.head())
print(df.describe())
print(df.info())

# Encode nominal features
df_encode = df.copy()
# Correlation Map
df_encode = df_encode.apply(lambda x: pd.factorize(x)[0])

# #Correlation Map after Removing Columns with more than 20% NULL values

# corr = df_encode.corr().round(2)
# corr.style.background_gradient(cmap='coolwarm')
# print(corr)
# corr_df = pd.DataFrame(corr)
# print(corr.max().sort_values(ascending=False))
# fig = plt.figure(figsize=(16,16))
# plt.title("Heatmap of Pearson's Correlation Coefficients Mushroom Dataset")
# sns.heatmap(corr, annot = True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.tight_layout()
# plt.show()


target = 'class'
cols = df_encode.columns
cols = cols.drop('class')
features = list(cols)

# =========================
# PCA Analysis
# =========================
Y = df_encode[target]
X = df_encode[features].values
X = StandardScaler().fit_transform(X)
pca = PCA(n_components='mle', svd_solver='full')
# pca = PCA(n_components = 2, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
print("Original Dim", X.shape)
print("Transformed Dim", X_PCA.shape)
print(f"explained variance ratio {pca.explained_variance_ratio_}")
plt.figure()
x = np.arange(1, len(pca.explained_variance_ratio_) + 1, 1)
plt.xticks(x)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.ylabel("Explained Variance")
plt.xlabel("Number of Components")
plt.title("PCA Analysis")
plt.show()
j = 0
for i in range(11):
    print(f'Variance explained by the {i+1} first principal component = {np.cumsum(pca.explained_variance_ratio_ * 100)[j]:.2f}')
    j += 1

print('*' * 50)

# ==========================================================
# SVD Analysis and Conditional Number on the original data
# ==========================================================

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'Original Data: Singular Values {d}')
print(f'Original Data: condition number {LA.cond(X):.2f}')
print('*' * 50)

# ==========================================================
# SVD Analysis and Conditional Number on the transformed data
# ==========================================================

H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'Transformed Data: Singular Values {d_PCA}')
print(f'Transformed Data: condition number {LA.cond(X_PCA):.2f}')
print('*' * 50)

# ==========================================================
# Construction of reduced dimension dataset
# ==========================================================
a, b = X_PCA.shape
column = []
for i in range(b):
    column.append(f'Principal Col {i + 1}')
df_PCA = pd.DataFrame(data=X_PCA, columns=column)
df_PCA = pd.concat([df_PCA, Y], axis=1)
print(df_PCA)
print('*' * 50)

# ********************************************************
#     Decision Tree
# ********************************************************
print(f'features = {features}')
X = df_encode[features]
y = df_encode[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
clf_model.fit(X_train, y_train)
DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
y_predict = clf_model.predict(X_test)


accuracy_score(y_test, y_predict)
target = list(df['class'].unique())
feature_names = list(X.columns)
# tree.plot_tree(clf_model);
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf_model,
               feature_names=feature_names,
               class_names=['poisonous', 'edible'],
               filled = True);


sort = clf_model.feature_importances_.argsort()
figure = plt.figure(figsize=(8,8))
plt.barh(feature_names, clf_model.feature_importances_[sort])
plt.xlabel("Importance")
plt.title("Features  Importance for Target Variable (Poisonous/Edible)")
plt.tight_layout()
fig.savefig('decision_tree.png')
plt.show()

#----------Drop the cap-diameter column to reduce multicollinearity-----------#
# df.drop('cap-diameter', axis = 1, inplace = True)

# ********************************************************
#     Mapping Categorical Data
# ********************************************************
# class is the target variable
dict_class = {'p': 'poisonous', 'e': 'edible'}
# cap shape dictionary
dict_cs = {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 's': 'sunken', 'p': 'spherical', 'o': 'others'}
dict_capcolor = {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red',
                 'w': 'white', 'y': 'yellow', 'l': 'blue', 'o': 'orange', 'k': 'black'}
dict_ga = {'a': 'adnate', 'x': 'adnexed', 'd': 'decurrent', 'e': 'free', 's': 'sinuate', 'p': 'pores', 'f': 'none',
           'u': 'unknown'}
dict_stemroot = {'b': 'bulbous', 's': 'swollen', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs',
                 'r': 'rooted'}
dict_gillcolor = dict_capcolor.copy()
dict_gillcolor['f'] = 'none'
dict_ringtype = {'c': 'cobwebby', 'e': 'evanescent', 'r': 'flaring', 'g': 'grooved', 'l': 'large', 'p': 'pendant',
                 's': 'sheathing', 'z': 'zone', 'y': 'scaly', 'm': 'movable', 'f': 'none', '?': 'unknown'}
dict_habitat = {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'h': 'heaths', 'u': 'urban', 'w': 'waste',
                'd': 'woods'}
dict_season = {'s': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'}
df['class'] = df['class'].map(dict_class)
df['cap-shape'] = df['cap-shape'].map(dict_cs)
df['cap-color'] = df['cap-color'].map(dict_capcolor)
df['does-bruise-or-bleed'] = df['does-bruise-or-bleed'].map({'t': 'True', 'f': 'False'})
df['gill-attachment'] = df['gill-attachment'].map(dict_ga)
df['gill-color'] = df['gill-color'].map(dict_gillcolor)
df['stem-color'] = df['stem-color'].map(dict_gillcolor)
df['has-ring'] = df['has-ring'].map({'t': 'True', 'f': 'False'})
df['ring-type'] = df['ring-type'].map(dict_ringtype)
df['habitat'] = df['habitat'].map(dict_habitat)
df['season'] = df['season'].map(dict_season)

mushroom = df.copy()

#==========================================================================
#          Outlier Detection and Removal (stem-height)
#==========================================================================

t = pd.DataFrame(mushroom['stem-height'])
median = np.median(t['stem-height'])
print(median)
lower = t[t['stem-height'] < median]
print(lower)
upper = t[t['stem-height'] > median]
print(upper)
Q1 = np.median(lower)
Q3 = np.median(upper)
print(f'Q1 for stem-height = {Q1}')
print(f'Q3 for stem-height = {Q3}')
print("")
IQR = Q3 -Q1
print(f'IQR = {IQR:.2f}')
low_outliers = Q1 - 1.5*IQR
high_outliers = Q3 + 1.5*IQR
print(f'Any stem-height value less than {low_outliers:.2f} or higher {high_outliers:.2f} than is an outlier.')
sns.boxplot(data = mushroom, y = 'stem-height')
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot for stem-height')
plt.show()
outliers_height = mushroom[(mushroom['stem-height'] > high_outliers) & (mushroom['stem-height'] < low_outliers)]
print(f'Number of outliers found for stem-height  = {len(outliers_height)}')
print('*' * 50)
#==========================================================================
#          Outlier Detection and Removal (stem-width)
#==========================================================================

t = pd.DataFrame(mushroom['stem-width'])
median = np.median(t['stem-width'])
print(median)
lower = t[t['stem-width'] < median]
print(lower)
upper = t[t['stem-width'] > median]
print(upper)
Q1 = np.median(lower)
Q3 = np.median(upper)
print(f'Q1 for stem-width = {Q1:.2f}')
print(f'Q3 for stem-width = {Q3:.2f}')
print("")
IQR = Q3 -Q1
print(f'IQR = {IQR:.2f}')
low_outliers = Q1 - 1.5*IQR
high_outliers = Q3 + 1.5*IQR
print(f'Any stem-width value less than {low_outliers:.2f} or higher {high_outliers:.2f} than is an outlier.')
sns.boxplot(data = mushroom, y = 'stem-width')
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot for stem-width')
plt.show()


outliers_width = mushroom[(mushroom['stem-width'] > high_outliers) & (mushroom['stem-width'] < low_outliers)]
print(f'Number of outliers found for stem-height  = {len(outliers_height)}')
outliers = [outliers_height, outliers_width]
print('*' * 50)

#==========================================================================
#          Outlier Detection and Removal (cap-diameter)
#==========================================================================

t = pd.DataFrame(mushroom['cap-diameter'])
median = np.median(t['cap-diameter'])
print(median)
lower = t[t['cap-diameter'] < median]
print(lower)
upper = t[t['cap-diameter'] > median]
print(upper)
Q1 = np.median(lower)
Q3 = np.median(upper)
print(f'Q1 for cap-diameter = {Q1:.2f}')
print(f'Q3 for cap-diameter = {Q3:.2f}')
print("")
IQR = Q3 -Q1
print(f'IQR = {IQR:.2f}')
low_outliers = Q1 - 1.5*IQR
high_outliers = Q3 + 1.5*IQR
print(f'Any cap-diameter value less than {low_outliers:.2f} or higher {high_outliers:.2f} than is an outlier.')
sns.boxplot(data = mushroom, y = 'cap-diameter')
# plt.boxplot(t)
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot for cap-diameter')
plt.show()


outliers_cap = mushroom[(mushroom['cap-diameter'] > high_outliers) & (mushroom['cap-diameter'] < low_outliers)]
print(f'Number of outliers found for cap-diameter  = {len(outliers_cap)}')
print('*' * 50)

#==========================================================================
#          Check Normality - Histogram
#==========================================================================
#
#
sns.histplot(data=mushroom, x='stem-height', bins=30)
plt.title('Histogram of Mushroom Stem-Height')
plt.show()

sns.histplot(data=mushroom, x='stem-width', bins=30)
plt.title('Histogram of Mushroom Stem-Width')
plt.show()


sns.histplot(data=mushroom, x='cap-diameter', bins=15)
plt.title('Histogram of Mushroom Cap-Diameter')
plt.show()
plt.close()



#==========================================================================
#         Normalize data - Create histograms
#==========================================================================
mushroom['stem-height-norm'] = stats.norm.ppf(stats.rankdata(mushroom['stem-height'])/(len(mushroom['stem-height']) + 1))
mushroom['stem-width-norm'] = stats.norm.ppf(stats.rankdata(mushroom['stem-width'])/(len(mushroom['stem-width']) + 1))
mushroom['cap-diameter-norm'] = stats.norm.ppf(stats.rankdata(mushroom['stem-height'])/(len(mushroom['stem-height']) + 1))
mushroom.to_csv("mushrooms.csv")

print("-"*80)
fig = plt.figure(figsize=(9,9))
fig.add_subplot(3, 2, 1)
plt.title('Histogram of Mushroom Stem-Height')
sns.histplot(data=mushroom, x='stem-height', bins=20, stat = 'percent' )
plt.xlabel('stem-height (cm)')
plt.grid(True)
fig.add_subplot(3, 2, 2)
plt.title('Histogram of Mushroom Stem-Height (Normalized)')
sns.histplot(data=mushroom, x='stem-height-norm', bins=20, stat = 'percent')
plt.xlabel('stem-height (cm)')
plt.grid(True)
fig.add_subplot(3, 2, 3)
plt.title('Histogram of Mushroom Stem-Width')
sns.histplot(data=mushroom, x='stem-width', bins=20, stat = 'percent')
plt.xlabel('stem-width (cm)')
fig.add_subplot(3, 2, 4)
plt.title('Histogram of Mushroom Stem-Width (Normalized)')
sns.histplot(data=mushroom, x='stem-width-norm', bins=20, stat = 'percent')
plt.xlabel('stem-width (cm)')
plt.tight_layout()
fig.add_subplot(3, 2, 5)
plt.title('Histogram of Mushroom Cap-Diameter')
sns.histplot(data=mushroom, x='cap-diameter', bins=20, stat = 'percent')
plt.xlabel('cap-diameter (cm)')
plt.tight_layout()
fig.add_subplot(3, 2, 6)
plt.title('Histogram of Mushroom Cap-Diameter (Normalized)')
sns.histplot(data=mushroom, x='cap-diameter-norm', bins=20, stat = 'percent')
plt.xlabel('cap-diameter (cm)')
plt.tight_layout()
plt.show()
plt.close()

#==========================================================================
#        QQ Plots - stem-width / stem-height
#==========================================================================
print("-"*80)
figure, axes = plt.subplots(1, 2, figsize=(10,8))
axes[0].set_title('Original: stem-width (cm) ')
sm.qqplot(mushroom['stem-width'], line='45', ax = axes[0])
axes[0].set_xlabel('stem-width (cm)')
axes[1].set_title('Normalized: stem-width (cm)')
sm.qqplot(mushroom['stem-width-norm'], line='45', ax = axes[1])
axes[1].set_xlabel('stem-width (cm)')
axes[0].grid()
axes[1].grid()
plt.tight_layout()
plt.show()

print("-"*80)
figure, axes = plt.subplots(1, 2, figsize=(10,10))
axes[0].set_title('Original: stem-height (cm)')
sm.qqplot(mushroom['stem-height'], line='45', ax = axes[0])
axes[1].set_title('Normalized: stem-height (cm)')
sm.qqplot(mushroom['stem-height-norm'], line='45', ax = axes[1])
axes[0].grid()
axes[1].grid()
plt.tight_layout()
plt.show()

print("-"*80)
figure, axes = plt.subplots(1, 2, figsize=(10,10))
axes[0].set_title('Original: cap-diameter (cm)')
sm.qqplot(mushroom['cap-diameter'], line='45', ax = axes[0])
axes[1].set_title('Normalized: cap-diameter (cm)')
sm.qqplot(mushroom['cap-diameter-norm'], line='45', ax = axes[1])
axes[0].grid()
axes[1].grid()
plt.tight_layout()
plt.show()


#==========================================================================
#  Statistical Normality Tests K-S, Shapiro (Before Normalization)
#==========================================================================
print("---------------------Before Normalization----------------------")
print("Normality Tests: stem-height")
x = mushroom['stem-height']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
shapiro_x = stats.shapiro(x)
# print(shapiro_x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))
fig = plt.figure(figsize=(8,8))

#==========================================================================
print("Normality Tests: stem-width")

x = mushroom['stem-width']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))
print("-"*80)
shapiro_x = stats.shapiro(x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))
print("-"*80)
#==========================================================================
print("Normality Tests: cap-diameter")
x = mushroom['cap-diameter']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))
print("-"*80)
shapiro_x = stats.shapiro(x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))
print("-"*80)


#==========================================================================
#  Statistical Normality Tests K-S, Shapiro
#==========================================================================
print("---------------------After Normalization----------------------")
print("Normality Tests: stem-height")
x = mushroom['stem-height-norm']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
shapiro_x = stats.shapiro(x)
# print(shapiro_x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))

#==========================================================================
print("Normality Tests: stem-width")
x = mushroom['stem-width-norm']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))
print("-"*80)
shapiro_x = stats.shapiro(x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))
print("-"*80)
#==========================================================================
print("Normality Tests: cap-diameter")
x = mushroom['cap-diameter-norm']
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.5f}".format(ks_x[1]))
print("-"*80)
shapiro_x = stats.shapiro(x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.5f}".format(ks_x[1]))

print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.5f}".format(da_k_x[1]))
print("-"*80)



#==========================================================================
#                   Data Visualizations
#==========================================================================
mushroom.drop('stem-width', axis= 1, inplace=True)
mushroom.drop('stem-height', axis= 1, inplace=True)
mushroom.drop('cap-diameter', axis= 1, inplace=True)

# Correlation Map after Removing non-normalized columns
df_encode = mushroom.copy()
#Correlation Map
df_encode = df_encode.apply(lambda x: pd.factorize(x)[0])
corr = df_encode.corr().round(2)
corr.style.background_gradient(cmap='coolwarm')
print(corr)
corr_df = pd.DataFrame(corr)
print(corr.max().sort_values(ascending=False))
fig = plt.figure(figsize=(24,24))
plt.title("Heatmap of Pearson's Correlation Coefficients Mushroom Dataset")
sns.heatmap(corr, annot = True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.set(font_scale = 1.8)
plt.tight_layout()
plt.show()

sns.set(font_scale = 1)
#-------------------------Pairplot------------------#
fig = plt.figure(figsize=(16,16))
plt.title('Pairplot of Quantitative Features by Class')
sns.pairplot(mushroom, hue = 'class')
plt.show()

# plt.title('Pairplot of All Features by Class')
# sns.pairplot(df_encode, hue = 'class')
# plt.show()

#-----------------------scatterplot----------------#
sns.scatterplot(data=mushroom, x='stem-width-norm', y='stem-height-norm', hue = 'class')
plt.title('Scatter plot of Mushroom Stem-Height vs. Stem-Width')
plt.ylabel('stem-height (cm)')
plt.xlabel('stem-width (cm)')
plt.show()

sns.scatterplot(data=mushroom, x='stem-width-norm', y='cap-diameter-norm', hue = 'class')
plt.ylabel('cap-diameter (cm)')
plt.xlabel('stem-width (cm)')
plt.title('Scatter plot of Mushroom Cap-Diameter vs. Stem-Width')
plt.show()

sns.scatterplot(data=mushroom, x='stem-height-norm', y='cap-diameter-norm', hue = 'class')
plt.ylabel('cap-diameter (cm)')
plt.xlabel('stem-Height (cm)')
plt.title('Scatter plot of Mushroom Cap-Diameter vs. Stem-Height')
plt.show()

#-------------scatterplot with regression line-----#
sns.regplot(data = mushroom, x = 'stem-width-norm', y='stem-height-norm', ci=None)
plt.title('Scatter plot of Mushroom Stem-Height vs. Stem-Width')
plt.ylabel('stem-height (cm)')
plt.xlabel('stem-width (cm)')
plt.show()
#==========================================================================
#                   Linear Regression
#==========================================================================

# model = LinearRegression()
# X = mushroom[['stem-width-norm']]
# y = mushroom['class']
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)
# model.fit(X_train, y_train)
#
# predictions = model.predict(X_test)
# r2 = r2_score(y_test, predictions)
# rmse = mean_squared_error(y_test, predictions, squared=False)
#
# print('The r2 is: ', r2)
# print('The rmse is: ', rmse)

#==========================================================================
#                   Distplots
#==========================================================================

#-----------------Displots--------------------------------#
sns.displot(mushroom, x='stem-width-norm', hue='class', multiple='dodge', bins=30, stat = 'percent')
plt.title('Distplot of Mushroom Stem-Width')
plt.xlabel('stem-width (cm)')
plt.tight_layout()
plt.show()

sns.displot(mushroom, x='stem-width-norm', hue='class', multiple='dodge', bins=30, stat = 'percent')
plt.title('Distplot of Mushroom Stem-Width')
plt.xlabel('stem-width (cm)')
plt.tight_layout()
plt.show()


sns.displot(data=mushroom, x="stem-height-norm", hue='class', stat='density', bins=15)
plt.title('Distribution Plot of Mushroom Stem-Height')
plt.xlabel('stem-height (cm)')
plt.tight_layout()
plt.show()

sns.displot(data=mushroom, x="stem-width-norm", hue='class', stat='density', bins=30)
plt.title('Distribution Plot of Mushroom Stem-Width')
plt.xlabel('stem-width (cm)')
plt.tight_layout()
plt.show()

sns.displot(mushroom, x='stem-height-norm', hue='class', kind='kde', multiple='stack')
plt.title('Distribution Plot of Mushroom Stem-Height')
plt.xlabel('stem-height (cm)')
plt.tight_layout()
plt.show()

sns.histplot(mushroom, x='stem-height-norm', hue='class', fill=True)
plt.title('Hist Plot of Mushroom Stem-Height')
plt.xlabel('stem-height (cm)')
plt.tight_layout()
plt.show()

#Countplot
sns.countplot(data=mushroom, x='cap-color', hue='class')
plt.title('Count Plot of Mushroom Cap-Color')
plt.show()

#Piechart
poisonous = mushroom['class'].value_counts()['poisonous']
edible = mushroom['class'].value_counts()['edible']

class_count = np.array([poisonous, edible])
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(class_count, labels=['Poisonous', 'Edible'], autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Poisonous versus Edible Mushrooms")
plt.show()

p = mushroom[mushroom['class'] == 'poisonous']
e = mushroom[mushroom['class'] == 'edible']

cap_color = p['cap-color'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(cap_color, labels=dict_capcolor.values(), autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Mushroom Cap Colors")
plt.legend(dict_capcolor.values(), loc='best')
plt.tight_layout()
plt.show()

cap_shape = mushroom['cap-shape'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(cap_shape, labels=dict_cs.values(), autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Mushroom Cap Shape")
plt.legend(dict_cs.values(), loc='best')
plt.tight_layout
plt.show()

stem_color = p['stem-color'].value_counts()
stem_list = list(stem_color.index)
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(stem_color, labels=stem_list, autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Mushroom Stem Colors")
plt.legend(dict_gillcolor.values(), loc='best')
plt.tight_layout()
plt.show()


sns.displot(data=mushroom, x="class", hue="habitat", col="season", bins=15)
plt.title('Distplot of Mushrooms Habitat and Season')
plt.tight_layout()
plt.show()

sns.kdeplot(data=mushroom, x="stem-width-norm", y="stem-height-norm", hue='class', fill=True)
plt.ylabel('stem-height (cm)')
plt.xlabel('stem-width (cm)')
plt.title('KDE Plot Stem-Height vs. Stem-Width')
plt.show()


sns.catplot(x="season", col="class", data=mushroom, kind="count", height=4, aspect=.7)
plt.title('Catplot Seasons')
plt.show()

sns.lineplot(x='cap-shape', y='stem-height-norm', data=mushroom, hue='class')
plt.title('Lineplot Edible/Poisonous Stem-Height vs Cap-Shape')
plt.ylabel('stem-height (cm)')
plt.show()

sns.lineplot(x='cap-shape', y='stem-height-norm', data=mushroom, hue='cap-color')
plt.ylabel('stem-height (cm)')
plt.title('Lineplot Cap-Color Stem-Height vs Cap-shape')
plt.show()

sns.lineplot(x='cap-color', y='stem-height-norm', data=mushroom, hue='class')
plt.ylabel('stem-height (cm)')
plt.title('Lineplot Edible/Poisonous Stem-Height vs Cap-Color')
plt.show()

sns.lineplot(x='gill-color', y='stem-height-norm', data=mushroom, hue='class')
plt.title('Lineplot Cap-Color Stem-Height vs Gill-Color')
plt.ylabel('stem-height (cm)')
plt.show()


#-----------Boxplot-------------------------------------#
plt.title("Boxplot Stem-Height / Season by Class")
sns.boxplot(x="season", y="stem-height-norm", hue="class", data = mushroom)
plt.ylabel('stem-height (cm)')
plt.show()

plt.title("Boxplot Stem-Width / Season by Class")
sns.boxplot(x="season", y="stem-width-norm", hue="class", data = mushroom)
plt.ylabel('stem-width (cm)')
plt.show()


#-----------Violinplot-------------------------------------#
sns.violinplot(x="season", y="stem-height-norm", data=mushroom)
plt.title("Violinplot Stem-Height by Season")
plt.ylabel('stem-height (cm)')
plt.show()

sns.violinplot(x="season", y="stem-width-norm", data=mushroom)
plt.title("Violinplot Stem-Width by Season")
plt.ylabel('stem-width (cm)')
plt.show()

#-----------Countplot-------------------------------------#
sns.countplot(x="habitat", data=mushroom)
plt.title("Mushroom Habitat Countplot")
plt.show()

sns.countplot(x = 'ring-type', data = mushroom)
plt.title("Ring-Type Countplot")
plt.show()

sns.catplot(x = 'class', col = 'gill-attachment', data = mushroom, kind='count', legend = True, legend_out = True,
            height = 3.5, aspect = 0.8, col_wrap=4)
plt.show()

# mushroom['class-code']= pd.factorize(mushroom['class'])[0]
# sns.barplot(x = 'has-ring', y = 'stem-color', hue= 'class', data = mushroom)
# plt.legend(loc='upper right')
# plt.title("Barplot Has-Ring vs Stem-Color by Class")
# plt.tight_layout()
# plt.show()

sns.barplot(x = 'has-ring', y = 'stem-height-norm', hue = 'class', data = mushroom)
sns.barplot(x = 'has-ring', y = 'stem-width-norm', hue = 'class', data = mushroom)
plt.legend(loc='upper right')
plt.title("Barplot Has-Ring vs Stem-Color by Class")
plt.tight_layout()
plt.show()

stc= pd.DataFrame(mushroom['stem-color'].value_counts())
cpc= pd.DataFrame(mushroom['cap-shape'].value_counts())
gcc = pd.DataFrame(mushroom['gill-color'].value_counts())
ccc = pd.DataFrame(mushroom['cap-color'].value_counts())
rtc = pd.DataFrame(mushroom['ring-type'].value_counts())
hc = pd.DataFrame(mushroom['habitat'].value_counts())

sns.barplot(x = stc.index,  y="stem-color", data = stc)
plt.title("Bar plot: Stem-Color")
plt.show()
sns.barplot(x = cpc.index,  y="cap-shape", data = cpc)
plt.title("Bar plot: Cap-Shape")
plt.show()
sns.barplot(x = gcc.index,  y="gill-color", data = gcc)
plt.title("Bar plot: Gill-Color")
plt.show()
sns.barplot(x = ccc.index,  y="cap-color", data = ccc)
plt.title("Bar plot: Cap-Color")
plt.show()

sc = mushroom[['stem-color', 'class']]
sc = pd.DataFrame(sc.groupby(['stem-color', 'class']).size().unstack(fill_value=0))
sc.reset_index()
sns.barplot(x = sc.index, y = 'edible',  data= sc)
sns.barplot(x = sc.index, y = 'poisonous',  data= sc)
plt.legend(loc='upper right')
plt.title("Stacked Barplot Has-Ring vs Stem-Height-Width by Class")
plt.tight_layout()
plt.show()

sc = mushroom[['cap-shape', 'class']]
sc = pd.DataFrame(sc.groupby(['cap-shape', 'class']).size().unstack(fill_value=0))
sc.reset_index()
sns.barplot(x = sc.index, y = 'edible',  data= sc)
sns.barplot(x = sc.index, y = 'poisonous',  data= sc)
plt.legend(loc='upper right')
plt.title("Stacked Barplot Cap-Shape by Class")
plt.tight_layout()
plt.show()

# sc = mushroom[['stem-height-norm', 'class']]
# sc = pd.DataFrame(sc.groupby(['stem-height-norm', 'class']).size().unstack(fill_value=0))
# sc.reset_index()
# sns.barplot(x = sc.index, y = 'edible',  data= sc)
# sns.barplot(x = sc.index, y = 'poisonous',  data= sc)
# plt.legend(loc='upper right')
# plt.title("Stacked Barplot Stem-Height by Class")
# plt.tight_layout()
# plt.show()

# sns.barplot(x = 'gill-color', y = 'class-code', hue = 'gill-attachment', data = mushroom)
# plt.legend(loc='upper right')
# plt.title("Barplot Class by Stem-Color and Has-Ring")
# plt.tight_layout()
# plt.show()