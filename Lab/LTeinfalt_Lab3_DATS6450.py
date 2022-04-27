#Lydia Teinfalt
#Lab 3
#DATS 6450 Spring '22
#02/23/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('seaborn-darkgrid')

df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/CONVENIENT_global_confirmed_cases.csv")
print(df.describe())
print(df.head(10))
print(df.tail())
print("Shape before dropping NA from dataframe")
print(f"rows = {df.shape[0]}")
print(f"columns = {df.shape[1]}")

#First row of dataframe is State/Province
df1 = df.iloc[0]
dfnull = df.iloc[1:-1].isnull().values.any()
if dfnull:
    df = df.dropna(df.iloc[1:-1].isnull(), axis=0, inplace=True)

columns = df.columns

df1 = df.copy()
dfnull = df.isnull().values
df1 = df1.loc[1:]
df1['Date'] = pd.to_datetime(df1['Country/Region'])
countries = ['China', "United Kingdom", "Germany", "Brazil", "India", "Italy"]

for i in countries:
    col_names = df.columns[df.columns.str.startswith(i)]
    idx = [df.columns.get_loc(col) for col in col_names]
    # print(idx)
    new_column= i+"_sum"
    first_index = idx[0]
    last_index = idx[-1]
# china 57:89
# united Kingdom 249:259


df1['China_sum'] = df1.iloc[:,57:89].astype(float).sum(axis=1)
df1['United Kingdom_sum'] = df1.iloc[:,249:259].astype(float).sum(axis=1)
#
# #us covid cases - Line plot
# fig, ax = plt.subplots(figsize=(12,10))
# ax.plot(df1['Date'], df1['US'], linewidth = 2, marker = '.')
# ax.grid()
# ax.set_ylabel("Confirmed COVID19 Cases")
# plt.xticks(rotation=70)
# ax.set_xlabel("Year")
# ax.set_title("US Confirmed COVID19 Cases")
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.grid()
# plt.tight_layout()
# plt.show()
#
# #Question 5 Global Confirmed COVID19 Cases
# #“United Kingdom”, “China”, ”Germany”, ”Brazil”, “India” and “Italy”
# fig, ax = plt.subplots(figsize=(12,10))
# ax.plot(df1['Date'], df1['United Kingdom_sum'], label = 'United Kingdom', linewidth = 3)
# ax.plot(df1['Date'], df1['China_sum'], label = 'China',linewidth = 3)
# ax.plot(df1['Date'], df1['US'], label = 'US',linewidth = 3)
# ax.plot(df1['Date'], df1['Germany'], label = 'Germany', linewidth = 3)
# ax.plot(df1['Date'], df1['Brazil'], label = 'Brazil', linewidth = 3)
# ax.plot(df1['Date'], df1['India'], label = 'India', linewidth = 3)
# ax.plot(df1['Date'], df1['Italy'], label = 'Italy', linewidth = 3)
# ax.grid()
# ax.set_ylabel("Confirmed COVID19 Cases")
# plt.legend()
# plt.xticks(rotation=70)
# ax.set_xlabel("Year")
# ax.set_title("Global Confirmed COVID19 Cases")
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.grid()
# plt.tight_layout()
# plt.show()
#
#
#
# #Question 6 Plot histogram for US Covid19 Cases
# fig, ax = plt.subplots(figsize=(12,10))
# us = df1['US']
# ax.hist(df1['US'], label = 'US')
# ax.set_ylabel("Frequency")
# ax.legend()
# ax.grid()
# ax.set_xlabel("Values")
# ax.set_title("US Confirmed COVID19 Cases Histogram")
# plt.grid()
# plt.show()
#
#
# #“United Kingdom”, “China”, ”Germany”, ”Brazil”, “India” and “Italy”
# fig, ax = plt.subplots(figsize=(18,12))
# plt.grid()
# ax = fig.add_subplot(2,3,1)
# ax.hist(df1['United Kingdom_sum'], label ='United Kingdom', color = 'green' )
# ax.set_title("UK Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# ax.legend()
# ax = fig.add_subplot(2,3,2)
# ax.hist(df1['China_sum'], label = 'China', color = 'red')
# ax.set_title("China Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# ax.legend()
# ax = fig.add_subplot(2,3,3)
# ax.hist(df1['Germany'], label = 'Germany', color = 'brown')
# ax.set_title("Germany Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# ax.legend()
# ax = fig.add_subplot(2,3,4)
# ax.hist(df1['Brazil'], label = 'Brazil', color = 'blue')
# ax.set_title("Brazil Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# ax.legend()
# ax = fig.add_subplot(2,3,5)
# ax.hist(df1['India'], label = 'India', color = 'orange')
# ax.set_title("India Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# ax.legend()
# ax = fig.add_subplot(2,3,6)
# ax.hist(df1['Italy'], label = 'Italy', color = 'purple')
# ax.set_title("Italy Confirmed COVID19 Cases Histogram")
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Values")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# #Which country (from the list above) has the highest mean, variance and median of # of COVID confirmed cases?
# df_mean = df1[['China_sum', 'United Kingdom_sum', 'US', 'Germany', 'Brazil', 'India', 'Italy']].mean()
# df_mean.columns = "Mean"
# df_mean = df_mean.sort_values()
# print("Mean Confirmed COVID19 Cases")
# print(df_mean)
# print("US had highest mean")
# print("India had highest mean")
#
# df_var = df1[['China_sum', 'United Kingdom_sum', 'US', 'Germany', 'Brazil', 'India', 'Italy']].var()
# df_var.columns = "Variance"
# df_var = df_var.sort_values()
# print("Variance Confirmed COVID19 Cases")
# print(df_var)
# print("US had highest variance")
# print("India had second highest variance")
#
# df_med = df1[['China_sum', 'United Kingdom_sum', 'US', 'Germany', 'Brazil', 'India', 'Italy']].median()
# df_med.columns = "Median"
# df_med = df_med.sort_values()
# print("Median Confirmed COVID19 Cases")
# print(df_med)
# print("US had highest median")
# print("Brazil had highest median")
#
# print("#"*80)
import seaborn as sns
df = sns.load_dataset('titanic')
df1 = df.dropna(how='any')
print(df1.head(5))
print(df1.describe())

columns = df1.columns
#print(df1)

print(f"rows = {df1.shape[0]}")
print(f"columns = {df1.shape[1]}")
print(df1)
df1 =df1[['sex','survived', 'pclass']]

male_count = df1['sex'].value_counts()['male']
female_count = df1['sex'].value_counts()['female']
gender_count = np.array([male_count, female_count])

def absolute_value(val):
    a  = np.round(val/100.*gender_count.sum(), 0)
    return a

fig, ax = plt.subplots(figsize=(8,8))
ax.pie(gender_count, labels=['Male', 'Female'], autopct=absolute_value)
ax.axis('square')
ax.set_title("Pie chart of total people on Titanic")
plt.show()

percent_male = len(df1[df1['sex'] == 'male'])/len(df1)
percent_female = len(df1[df1['sex'] == 'female'])/len(df1)
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([percent_male, percent_female], labels=['Male', 'Female'],autopct='%.1f%%' )
ax.axis('square')
ax.set_title("Pie chart of total people on Titanic in %")
plt.show()
#
#
# #Write a python program that plot the pie chart showing the percentage of males who survived versus the percentage of males who did not survive. The final answer should look like bellow.
percent_male_survived = len(df1[(df1['sex'] == 'male')&(df1['survived'] == 1)])/len(df1[df1['sex'] == 'male'])
percent_male_not_survived = len(df1[(df1['sex'] == 'male')&(df1['survived'] == 0)])/len(df1[df1['sex'] == 'male'])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([percent_male_survived, percent_male_not_survived ], labels=['Male Survived', 'Male Not Survived'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Pie Chart of Male Survival on Titanic in %")
plt.show()
#
#
#Write a python program that plot the pie chart showing the percentage of females who survived versus the percentage of males who did not survive. The final answer should look like bellow.
percent_female_survived = len(df1[(df1['sex'] == 'female')&(df1['survived'] == 1)])/len(df1[df1['sex'] == 'female'])
percent_female_not_survived = len(df1[(df1['sex'] == 'female')&(df1['survived'] == 0)])/len(df1[df1['sex'] == 'female'])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([percent_female_survived, percent_female_not_survived ], labels=['Female Survived', 'Female Not Survived'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Pie Chart of Female Survival on Titanic in %")
plt.tight_layout()
plt.show()


#Write a python program that plot the pie chart showing the percentage passengers with first class, second class and third-class tickets
#
first_class = len(df1[(df1['pclass'] == 1)])/len(df1)
second_class = len(df1[(df1['pclass'] == 2)])/len(df1)
third_class = len(df1[(df1['pclass'] == 3)])/len(df1)
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([first_class, second_class, third_class], labels=['Ticket Class 1', 'Ticket Class 2', 'Ticket Class 3'],explode = (.03, .3, .3),autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Pie Chart Passengers Based on Level")
plt.tight_layout()
plt.show()
#
#
# #Write a python program that plot the pie chart showing the percentage passengers with first class, second class and third-class tickets
#
first_class_survived = len(df1[(df1['pclass'] == 1)&(df1['survived'] == 1)])/len(df1[df1['survived'] == 1])
second_class_survived = len(df1[(df1['pclass'] == 2)&(df1['survived'] == 1)])/len(df1[df1['survived'] == 1])
third_class_survived = len(df1[(df1['pclass'] == 3)&(df1['survived'] == 1)])/len(df1[df1['survived'] == 1])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([first_class_survived, second_class_survived, third_class_survived], labels=['Ticket Class 1', 'Ticket Class 2', 'Ticket Class 3'],explode = (.03, .3, .3),autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Pie Chart Survival Rate Based on Ticket Class")
plt.tight_layout()
plt.show()
#
# #Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive with the first-class ticket category. The final answer should look like bellow.
first_class_survived = len(df1[(df1['pclass'] == 1)&(df1['survived'] == 1)])/len(df1[df1['pclass'] == 1])
first_class_not_survived = len(df1[(df1['pclass'] == 1)&(df1['survived'] == 0)])/len(df1[df1['pclass'] == 1])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([first_class_survived, first_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 1")
plt.tight_layout()
plt.show()

# Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive with the second-class ticket category. The final answer should look like bellow.
second_class_survived = len(df1[(df1['pclass'] == 2)&(df1['survived'] == 1)])/len(df1[df1['pclass'] == 2])
second_class_not_survived = len(df1[(df1['pclass'] == 2)&(df1['survived'] == 0)])/len(df1[df1['pclass'] == 2])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([second_class_survived, second_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 2")
plt.tight_layout()
plt.show()


#Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive in the third-class ticket category.
third_class_survived = len(df1[(df1['pclass'] == 3)&(df1['survived'] == 1)])/len(df1[df1['pclass'] == 3])
third_class_not_survived = len(df1[(df1['pclass'] == 3)&(df1['survived'] == 0)])/len(df1[df1['pclass'] == 3])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie([third_class_survived, third_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 3")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(16,8))
ax = fig.add_subplot(3,3,1)
ax.pie(gender_count, labels=['Male', 'Female'], autopct=absolute_value)
ax.axis('square')
ax.grid()
ax.set_title("Pie chart of total people on Titanic")
ax = fig.add_subplot(3,3,2)
ax.pie([percent_male, percent_female], labels=['Male', 'Female'],autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.set_title("Pie chart of total people on Titanic in %")
ax = fig.add_subplot(3,3,3)
ax.pie([percent_male_survived, percent_male_not_survived ], labels=['Male Survived', 'Male Not Survived'],autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Pie Chart of Male Survival on Titanic in %")
ax = fig.add_subplot(3,3,4)
ax.pie([percent_female_survived, percent_female_not_survived ], labels=['Female Survived', 'Female Not Survived'],autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Pie Chart of Female Survival on Titanic in %")
ax = fig.add_subplot(3,3,5)
ax.pie([first_class, second_class, third_class], labels=['Ticket Class 1', 'Ticket Class 2', 'Ticket Class 3'],explode = (.03, .3, .3),autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Pie Chart Passengers Based on Level")
ax = fig.add_subplot(3,3,6)
ax.pie([first_class_survived, second_class_survived, third_class_survived], labels=['Ticket Class 1', 'Ticket Class 2', 'Ticket Class 3'],explode = (.03, .3, .3),autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Pie Chart Survival Rate Based on Ticket Class")
ax = fig.add_subplot(3,3,7)
ax.pie([first_class_survived, first_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 1")
ax = fig.add_subplot(3,3,8)
ax.pie([second_class_survived, second_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.grid()
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 2")
ax = fig.add_subplot(3,3,9)
ax.pie([third_class_survived, third_class_not_survived], labels=['Survival Rate', 'Death Rate'],autopct='%.1f%%' )
ax.axis('square')
ax.legend()
ax.set_title("Survival & Death Rate: Ticket Class 3")
ax.grid()
plt.show()