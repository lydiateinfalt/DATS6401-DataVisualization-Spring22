#Lydia Teinfalt
#2/16/2020
#DATS 6401
#Exam 1
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')

df = sns.load_dataset("diamonds")
print(df)
a = df.isnull().sum()
print('null values')
print(a)

col_list = df.columns
print(col_list)
df.dropna(how='any', inplace=True)
cut = df['cut'].unique()
print(f'The list of diamond cuts in the diamond dataset are {cut}')

color = df['color'].unique()
print(f'The list of diamond colors in the diamond dataset are {color}')

clarity = df['clarity'].unique()
print(f'The list of diamond clarity in the diamond dataset are {clarity}')

df1 = df[['cut','price']]
cut = df1['cut'].unique()
price = df1['price']
df2 = df1.groupby('cut').sum()
print(df2)
#Horizontal Bar
#Horizontal Bar
plt.figure(figsize=(16,8))
plt.barh(df2.index,df2['price'])
plt.ylabel('Cut')
plt.xlabel('Sales ($)')
plt.title('Sales count per cut’')
plt.legend()
plt.show()


max_sales = df2['price'].sort_values(ascending=False)
high_val = max_sales[0]
print(f"The diamond with ideal cut has the maximum sales per count")
print(f"The diamond with fair cut has the minimum sales per count")

#################################################

# Color

df3 = df[['color','price']]
color = df['color'].unique()
price = df['price']
df4 = df3.groupby('color').sum()
print(df3)

#Horizontal Bar
plt.figure(figsize=(12,8))
plt.barh(df4.index,df4['price'])
plt.ylabel('Color')
plt.xlabel('Sales ($)')
plt.title('Sales count per color')
plt.legend()
plt.show()

print("The diamond with G color has the maximum sales per count.")
print("The diamond with J color has the minimum sales per count.")

############################################


# Clarity

df5 = df[['clarity','price']]
clarity = df['clarity'].unique()
price = df['price']
df6 = df5.groupby('clarity').sum()

#Horizontal Bar
plt.figure(figsize=(12,8))
plt.barh(df6.index,df6['price'])
plt.ylabel('clarity')
plt.xlabel('Sales Count ($)')
plt.title('Sales count per clarity')
plt.legend()
plt.show()

print("The diamond with SI1 clarity has the maximum sales per count.")
print("The diamond with I1 has the minimum sales per count.")

#Subplot

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1,3,1)
ax1.barh(df2.index,df2['price'])
ax1.set_ylabel('cut')
ax1.set_xlabel('sales count')
ax1.set_title('Sales count per cut')


ax1 = fig.add_subplot(1,3,2)
ax1.barh(df4.index,df4['price'])
ax1.set_ylabel('color')
ax1.set_xlabel('sales count')
ax1.set_title('Sales count per color')


ax1 = fig.add_subplot(1,3,3)
ax1.barh(df6.index,df6['price'])
ax1.set_ylabel('clarity')
ax1.set_xlabel('Sales Count ($)')
ax1.set_title('Sales count per clarity')
plt.tight_layout()
plt.show()

###########################################
fig, ax = plt.subplots(1,1)
ax.pie(df2['price'], labels = cut, explode = (.03, .03, .3, .03, .03), autopct='%1.3f%%')
ax.axis('square')
ax.set_title('Sales count per cut in %')
plt.show()

total=df2['price'].sum()
df2['sales_perc']=df2.price/total*100
print(f"The diamond with Ideal cut has the maximum sales per count with 35.13% sales count.")
print(f"The diamond with Fair cut has the minimum sales per count with 3.31% sales count.")


###########################################

fig, ax = plt.subplots(1,1)
ax.pie(df4['price'], labels = color, explode = (.03, .03, .3, .03, .03, .03, .03), autopct='%1.3f%%')
ax.axis('square')
ax.set_title('Sales count per color in %')
plt.show()

total=df4['price'].sum()
df4['sales_perc']=df4.price/total*100
print(f"The diamond with H color has the maximum sales per count with 21.29% sales count.")
print(f"The diamond with D color has the minimum sales per count with 7.05% sales count.")



###########################################
#Clarity
fig, ax = plt.subplots(1,1)
ax.pie(df6['price'], labels = clarity, explode = (.03, .03, .3, .03, .03, .03, .03, 0.03), autopct='%1.3f%%')
ax.axis('square')
ax.set_title('Sales count per clarity in %')
plt.show()

total6=df6['price'].sum()
df6['sales_perc']=df6.price/total*100
print(f"The diamond with VVSI clarity has the maximum sales per count with 24.61% sales count.")
print(f"The diamond with IF clarity has the minimum sales per count with 1.37% sales count.")


#Pie

#Subplot

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1,3,1)
ax1.pie(df2['price'], labels = cut, explode = (.03, .03, .3, .03, .03), autopct='%1.3f%%')
ax1.set_ylabel('cut')
ax1.set_xlabel('sales count')
ax1.set_title('Sales count per cut')


ax1 = fig.add_subplot(1,3,2)
ax1.pie(df4['price'], labels = color, explode = (.03, .03, .3, .03, .03, .03, .03), autopct='%1.3f%%')
ax1.set_ylabel('color')
ax1.set_xlabel('sales count')
ax1.set_title('Sales count per color')


ax1 = fig.add_subplot(1,3,3)
ax1.pie(df6['price'], labels = clarity, explode = (.03, .03, .3, .03, .03, .03, .03, 0.03), autopct='%1.3f%%')
ax1.set_ylabel('clarity')
ax1.set_xlabel('Sales Count ($)')
ax1.set_title('Sales count per clarity')
plt.tight_layout()
plt.show()

####################3


df_bonus = df[df['clarity'] == 'VS1']
df_mean = df_bonus.groupby('cut')
print(df_mean)
