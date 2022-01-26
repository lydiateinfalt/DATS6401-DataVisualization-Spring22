# DATS 6401: Lab 1 (Spring 22)
# Lydia Teinfalt
# 01/26/2022
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Q1-Using the NumPy package in python create a random variable x,
# normally distributed about the mean of zero and variance 1.
# Create a random variable y normally distributed about the mean of 5 and variance of 2.
# Number of samples for both x and y = 1000.
import pandas

print("-------------------------------------------------------------------------")
print("Q1: sing the NumPy package in python create a random variable x,")
print("normally distributed about the mean of zero and variance 1.")
print("Create a random variable y normally distributed about the mean of 5 and variance of 2.")
print("Number of samples for both x and y = 1000.")

print("Formula: standard deviation is equal to the square root of variance.")
N = 1000
xmean = 0
std_x = np.sqrt(1)

ymean = 5
std_y = np.sqrt(2)

# Random number with normal distribution
x = np.random.normal(xmean, std_x, N)
y = np.random.normal(ymean, std_y, N)

# Q2: Write a python program that calculate the Pearson’s correlation coefficient between
# two random variables x and y in question 1. Hint: You need to implement the following formula
print("-------------------------------------------------------------------------")
print("Q2: Write a python program that calculates the Pearson's correlation "
      "coefficient between the two random variables.")

def corr_coeff(x, y):
    diff_x = []
    diff_y = []
    r = []
    diff_x_squared = []
    diff_y_squared = []

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if len(x) == len(y):
        for i in range(len(x)):
            x_hat = (x[i] - mean_x)
            diff_x.append(x_hat)
            diff_x_squared.append(np.square(x_hat))
            y_hat = (y[i]- mean_y)
            r.append((x_hat * y_hat))
            diff_y_squared.append(np.square(y_hat))
            return (np.sum(r))/(np.sqrt(np.sum(diff_x_squared))*np.sqrt(np.sum(diff_y_squared)))


#x = [1, 2, 3, 4, 5]
#y=  [12, 14, 16, 18, 20]
corr = corr_coeff(x, y)
#print(f"The Pearson's correlation coefficient is {corr:.2f} ")
print("-------------------------------------------------------------------------")
print("Q3")
print(f"The sample mean of random variable x is {np.mean(x):.2f}")
print(f"The sample mean of random variable y is {np.mean(y):.2f}")
print(f"The sample variance of random variable x is {np.var(x):.2f}")
print(f"The sample variance of random variable y is {np.var(y):.2f}")
print(f"The sample Pearson's correlation coefficient between X and Y is {corr:.2f}")

print("-------------------------------------------------------------------------")
print("Q4: Line Plot of Variable X x Variable Y")

# line plot
plt.figure(figsize=(12, 8))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random variable X vs Random Variable Y")
plt.plot(x, 'c', label='x')
plt.plot(y, 'y', label='y')
plt.legend()
plt.grid()
plt.show()

print("-------------------------------------------------------------------------")
print("Q5: Histogram of Variable X x Variable Y")
# histogram
plt.figure(figsize=(12, 8))
plt.hist(x, bins=50, label='x', alpha=.5)
plt.hist(y, bins=50, label='y', alpha=.5)
plt.title("Random variable X vs Random Variable Y")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

print("-------------------------------------------------------------------------")
print("Q6: Using pandas package in python read the ‘tute1.csv’ dataset. "
      "The set is the timeseries dataset with Sales, AdBudget and GDP column.")


df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/tute1.csv", parse_dates=[0])
df.columns = ["Date", "Sales", "AdBudget", "GDP"]

date_dict = {"1-Mar": "Mar-01", "1-Jun": "Jun-01", "1-Sep": "Sep-01", "1-Dec": "Dec-01",
             "2-Mar": "Mar-02", "2-Jun": "Jun-02", "2-Sep": "Sep-02", "2-Dec": "Dec-02",
             "3-Mar": "Mar-03", "3-Jun": "Jun-03", "3-Sep": "Sep-03", "3-Dec": "Dec-03",
             "4-Mar": "Mar-03", "4-Jun": "Jun-04", "4-Sep": "Sep-04", "4-Dec": "Dec-04",
             "5-Mar": "Mar-03", "5-Jun": "Jun-05", "5-Sep": "Sep-05", "5-Dec": "Dec-05"}
df = df.replace({"Date": date_dict})
df['Date'] = pd.to_datetime(df['Date'], format="%b-%y").dt.strftime("%b-%Y")
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Year'] = pd.to_datetime(df['Date']).dt.year


print("-------------------------------------------------------------------------")
print("Q7: Find the Pearson's correlation coefficient between Sales, AdBudget, and GDP.")
print("-------------------------------------------------------------------------")
print("Q8: Display message on the console:")
print(f"8a.The sample Pearson’s correlation coefficient between Sales & AdBudget is:  {corr_coeff(df.Sales, df.AdBudget):.2f}")
print(f"8b.The sample Pearson’s correlation coefficient between Sales and GDP is: {corr_coeff(df.Sales, df.GDP):.2f}")
print(f"8c.The sample Pearson’s correlation coefficient between AdBudget and GDP is: {corr_coeff(df.AdBudget, df.GDP):.2f}")
print("-------------------------------------------------------------------------")
print("Q9: Display the line plot of Sales, AdBudget, and GDP versus time.")
# Line plot

plt.figure()
df.plot(x='Date', y=['Sales', 'AdBudget', 'GDP'])
plt.xlabel("Date")
plt.title("Quarterly Sales, Ad Budget, and GDP from 1981-2005")
plt.ylabel("Values")
plt.grid()
plt.show()



print("-------------------------------------------------------------------------")
print("Q10: Plot the histogram plot of Sales, AdBudget, and GDP on one plot.")
# histogram plot
plt.figure()
plt.hist(df['Sales'],color='b',alpha=0.3, label='Sales', histtype='stepfilled')
plt.hist(df['AdBudget'],color='g',alpha=0.3, label='Ad Budget', histtype='stepfilled')
plt.hist(df['GDP'],color='r',alpha=0.3, label='GDP', histtype='stepfilled')
plt.title("Frequency Histogram of Sales, Ad Budget, and GDP")
plt.legend()
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.grid()
plt.show()






