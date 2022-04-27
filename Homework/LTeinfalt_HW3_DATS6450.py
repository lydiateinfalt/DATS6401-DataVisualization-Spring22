# DATS 6401: Homework 3 (Spring 22)
# Lydia Teinfalt
# 03/09/2022
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
#
# #-----------------------------------------------------------------------------------------
# # 3. Using the seaborn package graph the histogram plot “flipper_length_mm”.
# # Use the ‘darkgrid’ style as seaborn the theme. Write down your observation about the graph on the console.
# sns.set_theme(style='darkgrid')
# sns.histplot(data = penguin,
#              x = 'flipper_length_mm')
# plt.show()
# print('#3 The penguins flipper_length_mm appears to have somewhat normal distribution ')
# print('with two peaks also known as bimodal distribution. ')
# print('One peak is approximately at 190-195 and the second peak approximately at 210.')
#
#4. Change the bin width in the previous question to 3 and replot the graph. Hint: binwidth


# sns.histplot(data = penguin,
#              x = 'flipper_length_mm',
#              binwidth=3)
# plt.show()
#
# #5. Change the bin numbers to 30 in the previous question and replot the graph. Hint: bins
# sns.histplot(data = penguin,
#              x = 'flipper_length_mm',
#              bins=30)
# plt.show()
#-----------------------------------------------------------------------------------------
#6. Using the seaborn “displot”, graph then histogram plot per the species.
# Hint: You need to use the ‘hue’ . Write down your observation about the graph on the console.

# sns.displot(data = penguin,
#             x = 'flipper_length_mm',
#             hue= 'species')
# plt.show()
# #
# print('#6 Each penguin species have specific range of flipper lengths. The adelie penguins')
# print('flipper lengths are shorter than the gentoo and the chinstrap are in the middle. ')
# print('The first peak of the bimodal distribution mentioned in #3 is by the adelie. ')
# print('The second peak is by the gentoo penguins. There are less number of chinstrap than other species.')
# #7. Re-graph the plot in the previous question with element=’step’.
# sns.histplot(penguin, x = "flipper_length_mm", hue = "species", element = "step")
# plt.show()
#-----------------------------------------------------------------------------------------
# 8. Using the seaborn package graph the ‘stacked’ histogram plot of ‘flipper_length_mm’ with
# respect to ‘species’. Hint: multiple = ‘stack’. Write down your observation about the graph on the
# console.

sns.histplot(data = penguin,
             x = 'flipper_length_mm',
             hue = 'species',
             multiple = 'stack')
plt.show()
#
print('#8 Compared to plot in number 6,the stacked option makes it easier to differentiate ')
print('flipper lengths of the three types of penguins. This plot makes it clear that the second')
print('peak of flipper length is of adelie penguins and not gentoo')
# 9.0  Using the seaborn package and ‘displot’, graph the histogram plot of ‘flipper_lebgth_mm’
# with respect to ‘sex’ and use the option “dodge”. Write down your observation about
# the graph on the console. Hint: multiple = ‘dodge’.

sns.displot(penguin,
            x = 'flipper_length_mm',
            hue = 'sex',
            multiple = 'dodge')
plt.show()

print('#9 Female penguins have the same or longer flipper lengths then')
print('the male penguins irrespective of species. ')
print('There is an exception to this at 175, 205, and 225 mm flipper lengths.')

#10. Using the seaborn package and ‘displot’, graph the histogram plot of ‘flipper_lebgth_mm’
# in two separate figures (not shared axis) but in one single graph (one row two columns).
# What is the most frequent range of flipper length in mm for male and female penguins?
sns.displot(data=penguin, x="flipper_length_mm", hue="species", col="sex")
plt.show()
print("#10 The range 180-210 mm appears to be the most frequent for male and female penguins. ")
# #11. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# # to species in one graph (shared axis) in a normalized fashion. Which species has the larger
# # flipper length and what is the approximate range? Hint: Use stat = ‘density’
sns.displot(data = penguin, x="flipper_length_mm", hue= 'species', stat = 'density')
plt.show()
print('#11 The gentoo has the larger flipper length.')
print('The approx. range is 210-235 mm. ')
#
# 12. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# to sex in one graph (shared axis) in a normalized fashion. Which sex has the larger flipper length
# and what is the approximate flipper length? Hint: Use stat = ‘density’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', stat = 'density')
plt.show()
print("#12 Female penguins have longer flipper in the lengths 180-195 and 210-215 mm.")
print("#Males have longer flippers in the range above 195 and less than 210, and higher than 230.")
# 13. Using the seaborn package compare the distribution of ‘flipper_length_mm’ with respect
# to species in one graph (shared axis) in a normalized fashion that the bars height sum to 1.
# Which flipper length and species is more probable ? Hint: Use stat = ‘probability’
sns.displot(penguin, x = 'flipper_length_mm', hue = 'species', stat = 'probability')
plt.show()
print("#13 Adelie penguins most probable to have flipper in the lengths 170-195, Chinstrap having flippers in the")
print ("range of 180- 210 mm. Gentoo penguins most probable when flipper lengths are 210-230 mm. ")

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
print("")
print("This graph gives a clearer picture of the distinct normal distributions of three penguin species with ")
print("the majority of Adelie having 180 – 200 mm flipper lengths. Adelie also represent the highest number of ")
print("samples in the dataset. Chinstrap has the least number of samples in the dataset and majority of them ")
print("having 190-210 mm flipper lengths.  There is an overlap between the Adelie and Chinstrap distributions. ")
print("Majority of gentoo penguins having flipper lengths of 210-225 mm with the peak at approximately 215 mm.")

# 19. Repeat question 15 with argument fill = True. Write down your observations about the graph.
sns.displot(penguin, x = 'flipper_length_mm', hue = 'sex', kind='kde', fill = True)
plt.show()
print("")
print("#19 Both male and female penguins have bimodal distributions. Peaks for female penguins are ")
print("at approx. 190 and 210mm.  Peaks for the male penguins are at 195 and 225 mm.  ")
# 20. Plot the scatter plot and the regression line in one graph for the
# x-axis is ‘bill_length_mm’ and y-axis is ‘bill_depth_mm’.
# How the ‘bill_length_mm’ and ‘bill_depth_mm’ are correlated?
sns.scatterplot(data = penguin, x = 'bill_length_mm', y = 'bill_depth_mm')
plt.show()
print("#20 Between 35-40 mm bill length, there is a tendency between 16-20mm bill depth. ")
print("Between 45-50 mm bill length, there is a tendency between 14-16mm bill depth.")
print("Minority case that between 49-52mm, there is a tendency with 18-20 mm bill depth.")
print("Suggesting that longer bill length is more like to have shorter bill depth. ")
print("Suggesting that shorter length is more like to have longer bill depth. ")
print("With the exception of the a smaller group of penguins having both long bill lengths and depths.")

# 21. Using the count plot, display the bar plot of the number penguins in different
# islands using the hue = species. Write down your observations about the graph?
sns.countplot(data = penguin, x = 'island', hue = 'species')
plt.show()
print("Only the Adelie penguins inhabit Torgersen island. But Adelie penguins live on the other islands as well. ")
print("Gentoos live only on Briscoe island along with Adelie species but Gentoos are the dominant species.")
print("Chinstraps live only on Dream island along with Adelie species but Chinstraps are the dominant species. ")
#
# 22. Using the count plot, display the bar plot of the number of male and
# female penguins [in the dataset] using the hue = species. Write down your observations about the graph?
sns.countplot(data = penguin, x = 'sex', hue = 'species')
plt.show()
print("#22 The number of female and male penguins are evenly divided for all species.")
print("Most penguins are Adelie and that is reflected the count plots.")
print("Second most popular type of penguins is the gentoo -- slightly more males then female.")
print("Least populous species is the chinstrap but evenly divided between male and female.")

sns.set_theme(style="dark")
# 23. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘bill_depth_mm’ for
# male and female.
sns.displot(data = penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()
#
# 24. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
sns.displot(data = penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()
#
# 25. Plot the bivariate distribution between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female.
sns.displot(data = penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex', kind = 'kde', fill = True)
plt.show()
#
# 26. Using subplot, plot the last 3 questions in one graph with 3 rows and 1 column. Figure size = (8,16).
# Write down your observations about the plot in the last 3 questions.
fig, axes = plt.subplots(3, 1, figsize=(8, 16))
sns.kdeplot(data = penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex', fill = True, ax = axes[0])
sns.kdeplot(data = penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex', fill = True, ax = axes[1])
sns.kdeplot(data = penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex', fill = True, ax = axes[2])
plt.show()
print("#26 Most female penguins have lower bill lengths and depths then the males.")
print("For both sexes, there are two trends -- medium bill length with low bill depth. ")
print("The second trend is low bill length and high depth. ")
#
# # 27. Graph the bivariate distributions between “bill_length_mm” versus “bill_depth_mm” for male and female.
sns.displot(penguin, x="bill_length_mm", y="bill_depth_mm", hue= 'sex')
plt.show()
#
# # 28. Graph the bivariate distributions between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
sns.displot(penguin, x="bill_length_mm", y="flipper_length_mm", hue= 'sex')
plt.show()
#
# # 29. Graph the bivariate distributions between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female. Final plot like question 27.
sns.displot(penguin, x="flipper_length_mm", y="bill_depth_mm", hue= 'sex')
plt.show()



