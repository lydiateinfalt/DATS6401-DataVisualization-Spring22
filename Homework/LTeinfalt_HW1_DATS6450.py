# DATS 6401: Homework 1 (Spring 22)
# Lydia Teinfalt
# 02/03/2022

import seaborn as sns
import pandas as pd
import numpy as np
#1. Using the python pandas package load the titanic.csv dataset.
# Write a python program that reads all the column title and save them under variable ‘col’.
# The list of features in the dataset and the explanation is as followed:
df = sns.load_dataset("titanic")
col = df. columns
print(f'Col = {col}')

# Cntl + Shift + I to view method definition

df.describe()
#shape returns (#rows, # columns)
# Columns
print("Original number of columns=", df.shape[1])
# Rows
print("Original number of rows= ", df.shape[0])

#2. the titanic dataset needs to be cleaned due to nan entries. Remove all the nan in teh dataset.
print("Question 2")
df1 = df.copy()
df1.dropna(how='any', inplace = True)
print(df1.describe())
print(df1.head())

#3 The titanic dataset needs to be cleaned due to nan entries. Replace all the nan in the dataset with the mean of the column.
#Using the .describe() and .head() display the cleaned data.
print("Question 3")
df3 = df[['survived', 'pclass', 'sex', 'age']]
mean_age = df3['age'].mean()
pd.options.mode.chained_assignment = None
df3['age'].fillna(mean_age, inplace=True)
numeric_cols = df.select_dtypes(include='number').columns.to_list()
print(f'Numeric columns in the titanic dataframe {numeric_cols}')
df_copy = df.copy()
for num in numeric_cols:
      df_copy[num].fillna(df_copy[num].mean(), inplace=True)
print(df_copy.describe())
print(df_copy.head())
#print(df3.describe())
#print(df3.head())

#4 The titanic dataset needs to be cleaned due to nan entries. Replace all the nan in the dataset with the median of the column.
#Using the .describe() and .head() display the cleaned data.
print("Question 4")
df4 = df[['survived', 'pclass', 'sex', 'age']]
median_age = df4['age'].median()
df4['age'].fillna(median_age, inplace=True)
df_median = df.copy()
for num in numeric_cols:
      df_median[num].fillna(df_median[num].mean(), inplace=True)
print(df_median.describe())
print(df_median.head())

#5. Using pandas in python write a program that finds the total number of passengers on board.
#Then finds the numbers of males and females on board and diplay the following message:
print("Question 5")
print(f'a. The total number of passengers on board was df3 {len(df3)} ')
df_female = df3[df3['sex'] == 'female']
df_male = df3[df3['sex'] == 'male']
print(f'b. From the total number of passengers onboard, there were {(len(df_male)/len(df3))*100:.2f}% male passengers onboard.')
print(f'b. From the total number of passengers onboard, there were {(len(df_female)/len(df3))*100:.2f}% female passengers onboard.')
print("-"*80)

#6. Using pandas in python write a program that find out the number of survivals. Also, the total number of
#male survivals and total number of female survivals. Not: You need to use the Boolean & condition to filter the Dataframe. Then display
# a message on the console as follows:
print("Question 6")
df_survived = df3[df3['survived'] == 1]
df_survived_males = df3[(df3['survived'] == 1) & (df3['sex'] == 'male')]
df_survived_female = df3[(df3['survived'] == 1) & (df3['sex'] == 'female')]
print(f'a. Total number of survivals onboard was {(len(df_survived))}')
print(f'b. Out of {len(df_male)} male passengers onboard only {(len(df_survived_males)/len(df_male))*100:.2f}% male was survived.')
print(f'c. Out of {len(df_female)} male passengers onboard only {(len(df_survived_female)/len(df_female))*100:.2f}% male was survived.')
print("-"*80)

#7. Using pandas in python write a program that find out the number of passengers with upper class ticket. Then, find out the total number of male and
#female survivals with upper class ticket. Then display the following messages on the console:
print("Question 7")
df_upper = df3[df3['pclass'] == 1]
df_upper_survived = df3[(df3['pclass'] == 1)&(df3['survived'] == 1)]
df_upper_male = df3[(df3['pclass'] == 1)&(df3['sex'] == 'male')]
df_upper_female = df3[(df3['pclass'] == 1)&(df3['sex'] == 'female')]
df_male_survived = df3[(df3['pclass'] == 1)&(df3['survived'] == 1)&(df3['sex'] == 'male')]
df_female_survived = df3[(df3['pclass'] == 1)&(df3['survived'] == 1)&(df3['sex'] == 'female')]
print(f'a. There were a total number of {len(df_upper)} passengers with upper class ticket and only'
      f' {(len(df_upper_survived)/len(df_upper))*100 : .2f}% were survived.')
print(f'b. Out of {len(df_upper)} passengers with upper class ticket,{(len(df_upper_male)/len(df_upper))*100 : .2f}% passengers were male.')
print(f'c. Out of {len(df_upper)} passengers with upper class ticket,{(len(df_upper_female)/len(df_upper))*100 : .2f}% passengers were female.')
print(f'd. Out of {len(df_upper)} passengers with upper class ticket,{(len(df_male_survived)/len(df_upper_male))*100 : .2f}% male passengers survived.')
print(f'd. Out of {len(df_upper)} passengers with upper class ticket,{(len(df_female_survived)/len(df_upper_female))*100 : .2f}% female passengers survived.')
print("-"*80)

print("Question 8")
df3['Above50&Male'] = np.where((df3['age']>50)&(df3['sex']== 'male'), 'Yes','No')
df_Above50Male = df3[df3['Above50&Male'] == 'Yes']
print(df_Above50Male.head(5))
print(f'There were total number of {len(df_Above50Male)} male passengers above 50 years old.')

print("-"*80)
print("Question 9")
df3['Above50&Male&Survived'] = np.where((df3['age']>50)&(df3['sex']== 'male')&(df3['survived'] == 1), 'Yes','No')
df_9 = df3[df3['Above50&Male&Survived'] == 'Yes']
print(df_9.head(5))
print(f'There were total number of {len(df_9)} male passengers above 50 years old and survived.')
#Find the survival percentage rate of male passengers onboard who are above 50 years old?
print(f'The survival percentage rate of male passengers onboard who are above 50 years old was {(len(df_9)/len(df_male))*100:.2f}%')

print("-"*80)
print("Question 10")
df3['Above50&Female'] = np.where((df3['age']>50)&(df3['sex']== 'female'), 'Yes','No')
df_10a = df3[df3['Above50&Female'] == 'Yes']
print(df_10a.head(5))
print(f'There were total number of {len(df_10a)} female passengers above 50 years old.')



df3['Above50&Female&Survived'] = np.where((df3['age']>50)&(df3['sex']== 'female')&(df3['survived'] == 1), 'Yes','No')
df_10b = df3[df3['Above50&Female&Survived'] == 'Yes']
print(df_10b.head(5))
print(f'There were total number of {len(df_10b)} female passengers above 50 years old and survived.')
#Find the survival percentage rate of female passengers onboard who are above 50 years old?
print(f'The survival percentage rate of female passengers onboard who are above 50 years old was {(len(df_10b)/len(df_female))*100:.2f}%')


