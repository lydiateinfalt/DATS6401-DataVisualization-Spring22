# DATS 6401: Homework 3 (Spring 22)
# Lydia Teinfalt
# 02/03/2022
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = sns.load_dataset('penguins')
print("1. Load the ‘penguins’ dataset from the seaborn package. Display the last 5 observations. Display the dataset statistics. Hint: describe()")
print(data.tail(5))
print(data.describe())

#2. Dataset cleaning: Write a python program that check if the dataset is cleaned.
# If not, removed the missing observations from the data set.
# Display the portion of the code that perform the task here.
# Display the results that confirms the dataset is clean.

def clean_data(df):
    dfnull = df.iloc[1:-1].isnull().values.any()
    if dfnull:
        print("Cleaning up data")
        df1 = df.dropna(how='any')
    else:
        print(f"List any values that are null {df.isnull.values.any()}")
    return df1

print(data)
penguin = clean_data(data)
print("Post clean dataset")
print(penguin)

#-----------------------------------------------------------------------------------------
# 3. Using the seaborn package graph the histogram plot “flipper_length_mm”.
# Use the ‘darkgrid’ style as seaborn the theme. Write down your observation about the graph on the console.
sns.set_theme(style='darkgrid')
sns.histplot(data = penguin,
             x = 'flipper_length_mm')
plt.show()

#4. Change the bin width in the previous question to 3 and replot the graph. Hint: binwidth


sns.histplot(data = penguin,
             x = 'flipper_length_mm',
             binwidth=3)
plt.show()

#5. Change the bin numbers to 30 in the previous question and replot the graph. Hint: bins
sns.histplot(data = penguin,
             x = 'flipper_length_mm',
             bins=30)
plt.show()
#-----------------------------------------------------------------------------------------
#6. Using the seaborn “displot”, graph then histogram plot per the species.
# Hint: You need to use the ‘hue’ . Write down your observation about the graph on the console.

sns.displot(data = penguin,
            x = 'flipper_length_mm',
            hue= 'species')
plt.show()

#7. Re-graph the plot in the previous question with element=’step’.
sns.histplot(penguin, x = "flipper_length_mm", hue = "species", element = "step")
plt.show()
#-----------------------------------------------------------------------------------------
# 8. Using the seaborn package graph the ‘stacked’ histogram plot of ‘flipper_length_mm’ with
# respect to ‘species’. Hint: multiple = ‘stack’. Write down your observation about the graph on the
# console.

sns.histplot(data = penguin,
             x = 'flipper_length_mm',
             hue = 'species',
             multiple = 'stack')
plt.show()


# 9.0Using the seaborn package and ‘displot’, graph the histogram plot of ‘flipper_lebgth_mm’
# with respect to ‘sex’ and use the option “dodge”. Write down your observation about
# the graph on the console. Hint: multiple = ‘dodge’.

sns.displot(penguin,
            x = 'flipper_length_mm',
            hue = 'sex',
            multiple = 'dodge')
plt.show()

#10. Using the seaborn package and ‘displot’, graph the histogram plot of ‘flipper_lebgth_mm’
# in two separate figures (not shared axis) but in one single graph (one row two columns).
# What is the most frequent range of flipper length in mm for male and female penguins?
sns.displot(data=penguin, x="flipper_length_mm", hue="species", col="sex")
plt.show()

#11. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# to species in one graph (shared axis) in a normalized fashion. Which species has the larger
# flipper length and what is the approximate range? Hint: Use stat = ‘density’
sns.displot(data = penguin, x="flipper_length_mm", hue= 'species', stat = 'density')
plt.show()

# 12. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# to sex in one graph (shared axis) in a normalized fashion. Which sex has the larger flipper length
# and what is the approximate flipper length? Hint: Use stat = ‘density’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', stat = 'density')
plt.show()

# 13. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# to species in one graph (shared axis) in a normalized fashion that the bars height sum to 1.
# Which flipper length and species is more probable ? Hint: Use stat = ‘probability’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', stat = 'probability')
plt.show()

# 14. Using the seaborn package estimate the underlying density function of flipper length with respect
# to ‘species’ and the kernel density estimation. Plot the result. Hint: hue = ‘species’, kind = ‘kde’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'species', kind='kde')
plt.show()

# 15. Using the seaborn package estimate the underlying density function of flipper length with respect
# to ‘sex’ and the kernel density estimation. Plot the result. Hint: hue = ‘sex’, kind = ‘kde’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', kind='kde')
plt.show()

# 16. Repeat question 14 with argument multiple = ‘stack’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'species', kind='kde', multiple = 'stack')
plt.show()

# 17. Repeat question 15 with argument multiple = ‘stack’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', kind='kde', multiple = 'stack')
plt.show()
# 18. Repeat question 14 with argument fill = True. Write down your observations about the graph.
sns.displot(penguin, x = 'flipper_length_mm', hue = 'species', kind='kde', fill = True)
plt.show()

# 19. Repeat question 15 with argument fill = True. Write down your observations about the graph.
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', kind='kde', fill = True)
plt.show()

# 20. Plot the scatter plot and the regression line in one graph for the
# x-axis is ‘bill_length_mm’ and y-axis is ‘bill_depth_mm’.
# How the ‘bill_length_mm’ and ‘bill_depth_mm’ are correlated?
sns.scatterplot(data = penguin, x = 'bill_length_mm', y = 'bill_depth_mm')
plt.show()

# 21. Using the count plot, display the bar plot of the number penguins in different
# islands using the hue = species. Write down your observations about the graph?
sns.countplot(data = penguin, x = 'island', hue = 'species')
plt.show()

# 22. Using the count plot, display the bar plot of the number of male and
# female penguins [in the dataset] using the hue = species. Write down your observations about the graph?
sns.countplot(data = penguin, x = 'sex', hue = 'species')
plt.show()
sns.set_theme(style="dark")
# 23. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘bill_depth_mm’ for
# male and female.
sns.displot(data = penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()

# 24. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
sns.displot(data = penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()

# 25. Plot the bivariate distribution between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female.
sns.displot(data = penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()

# 26. Using subplot, plot the last 3 questions in one graph with 3 rows and 1 column. Figure size = (8,16).
# Write down your observations about the plot in the last 3 questions.
fig, axes = plt.subplots(3, 1, figsize=(8, 16))
sns.kdeplot(data = penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex', fill = True, ax = axes[0])
sns.kdeplot(data = penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex', fill = True, ax = axes[1])
sns.kdeplot(data = penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex', fill = True, ax = axes[2])
plt.show()

# 27. Graph the bivariate distributions between “bill_length_mm” versus “bill_depth_mm” for male and female.
sns.displot(penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex')
plt.show()

# 28. Graph the bivariate distributions between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
sns.displot(penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex')
plt.show()

# 29. Graph the bivariate distributions between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female. Final plot like question 27.
sns.displot(penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex')
plt.show()