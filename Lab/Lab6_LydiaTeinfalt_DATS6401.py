#Lydia Teinfalt
#Lab 6
#DATS 6401
#04/14/2022

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
# In this Homework, you will learn how to convert non-gaussian distributed dataset into a gaussian distributed dataset.
# 1. Generate a random data (x) with 5000 samples and normal distribution (mean = 0, std = 1).
# Then use np.cumsum to convert the generated normal data into a non-normal distributed data (y).
# Graph the normal (x) and non-normal (y) data set versus number of samples and histogram plot of the normal(x) and non-normal (y)
# dataset on a 2x2 figure using subplot. Number of bins = 100. Figure size = 9,7.
# Add grid and appropriate x-label, y-label, and title to the graph.
# ----------------------------------------------------------------------------------------------------------------------
x = np.random.normal(loc=0,scale=1, size=5000)
n = np.arange(0,5000)
y = np.cumsum(x)
fig = plt.figure(figsize=(9,7))
fig.add_subplot(2, 2, 1)
plt.plot(n,x)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Gaussian Data (x)')

fig.add_subplot(2, 2, 2)
plt.plot(n,y)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Non-Gaussian Data (y)')

fig.add_subplot(2, 2, 3)
plt.hist(x)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Histogram of Gaussian Data (x)')

fig.add_subplot(2, 2, 4)
plt.hist(y)
plt.grid(True)
plt.title('Histogram of Non-Gaussian data (y)')
plt.xlabel('# of samples')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 2. Perform a K-S Normality test on the x and y dataset [dataset generated in the previous question]. Display the p-value and statistics of the test for the x and y [a separate test is needed for x and y]. Interpret the K-S test [Normal or Not Normal with 99% accuracy]
# by looking at the p-value. Display the following information on the console:
print("-"*80)
print("Q2")
ks_x = stats.kstest(x, 'norm')
print(ks_x)
print("K-S test: statistics = {:.2f}".format(ks_x[0]) + " p_value = {:.2f}".format(ks_x[1]))
print(f'K-S test: x dataset looks normal.')

ks_y = stats.kstest(y,'norm')
print("K-S test: statistics = {:.2f}".format(ks_y[0]) + " p_value = {:.2f}".format(ks_y[1]))
print(f'K-S test: y dataset does not look normal.')
# ----------------------------------------------------------------------------------------------------------------------
# 3. Repeat Question 2 with the “Shapiro test”.
print("-"*80)
shapiro_x = stats.shapiro(x)
# print(shapiro_x)
print("Shapiro test: statistics = {:.2f}".format(shapiro_x[0]) + " p_value = {:.2f}".format(ks_x[1]))
print(f'Shapiro test: x dataset looks normal.')

shapiro_y = stats.shapiro(y)
print("Shapiro test: statistics = {:.2f}".format(shapiro_y[0]) + " p_value = {:.2f}".format(shapiro_y[1]))
print(f'Shapiro test: y dataset does not look normal.')
# ----------------------------------------------------------------------------------------------------------------------
# 4 Repeat Question 2 with the "D'Agostino's 𝐾2 test"
print("-"*80)
da_k_x = normaltest(x)
print("da_k_squared test: statistics = {:.2f}".format(da_k_x[0]) + " p_value = {:.2f}".format(da_k_x[1]))
print(f'da_k_squared test: x dataset looks normal.')

da_k_y = normaltest(y)
print("da_k_squared test: statistics = {:.2f}".format(da_k_y[0]) + " p_value = {:.2f}".format(da_k_y[1]))
print(f'da_k_squared test: y dataset does not look normal.')
# ----------------------------------------------------------------------------------------------------------------------
print("-"*80)
newY = stats.norm.ppf(stats.rankdata(y)/(len(y) + 1))
print(newY)
fig = plt.figure(figsize=(9,7))
fig.add_subplot(2, 2, 1)
plt.plot(n,y)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Non-Gaussian Data (y)')

fig.add_subplot(2, 2, 2)
plt.plot(n,newY)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Transformed Data - Gaussian (y hat)')

fig.add_subplot(2, 2, 3)
plt.hist(y)
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('# of samples')
plt.title('Histogram of Non-Gaussian Data (y)')

fig.add_subplot(2, 2, 4)
plt.hist(newY)
plt.grid(True)
plt.title('Histogram of Transformed Data - Gaussian (y hat)')
plt.xlabel('# of samples')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
print("-"*80)
figure, axes = plt.subplots(1, 2, figsize=(10,8))
plt.title("y data: Non-normal")
sm.qqplot(y, line='45', ax = axes[0])
plt.title("y hat data: Normal")
sm.qqplot(newY, line='45', ax = axes[1])
axes[0].grid()
axes[1].grid()
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
#7 Perform a K-S Normality test on the y transformed dataset. Display the p-value and statistics of the test for y
# transformed. Interpret the K-S test [Normal or Not Normal with 99% accuracy] by looking at the p-value. Display the
# following information on the console:
print("-"*80)
ks_newy = stats.kstest(newY, 'norm')
print("K-S test: statistics = {:.2f}".format(ks_newy[0]) + " p_value = {:.2f}".format(ks_newy[1]))
print(f'K-S test: y transformed dataset looks normal.')
# ----------------------------------------------------------------------------------------------------------------------
# Repeat Question 7 with the “Shapiro test”
print("-"*80)
shapiro_newy = stats.shapiro(newY)
print("Shapiro test: statistics = {:.2f}".format(shapiro_newy[0]) + " p_value = {:.2f}".format(shapiro_newy[1]))
print(f'Shapiro test: y transformed dataset looks normal.')
# ----------------------------------------------------------------------------------------------------------------------
# Repeat question 7 with the "D'Agostino's 𝐾2 test”
print("-"*80)
da_k_newy = normaltest(newY)
print("da_k_squared test: statistics = {:.2f}".format(da_k_newy[0]) + " p_value = {:.2f}".format(da_k_newy[1]))
print(f'da_k_squared test: y transformed dataset looks normal.')