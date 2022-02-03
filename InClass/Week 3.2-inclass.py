import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

df = sn.load_dataset('titanic')
survived_status = df.iloc[:,0]
df_male = df[df['sex'] == 'male']
df_female = df[df['sex'] == 'female']
df.dropna(how='any', inplace = True)
df_50 = df[df['age'] > 50]
print(f"Male passengers { len(df_male)/len(df): .2f}%")
print(f"Female passengers { len(df_female)/len(df): .2f}%")
print('#'*50)
print(f"Passengers above 50 { len(df_50)/len(df): .2f}%")
df_female_survived = df[(df['sex'] == 'female')&(df['survived'] == 1)]
df_male_survived = df[(df['sex'] == 'male')&(df['survived'] == 1)]
col_name = df.columns
df['age_status'] = np.where(df['age'] < 18, 'Teen', 'Adult')

print(f"Survived Female passengers { len(df_female_survived)/len(df_female): .2f}%")
print(f"Survived Male passengers { len(df_male_survived)/len(df_male): .2f}%")

df.drop(['embark_town', 'alive', 'alone', 'who', 'class', 'embarked'], axis=1, inplace=True)

df.rename({'age_status' : 'passenger_info'}, axis=1, inplace=True)

print(df)
df.insert(4,'age_mean', df['age']/df['age'].mean())
print(df)


df1 = sn.load_dataset('diamonds')
index = df[df['color']=='E'].index
df3 = df.drop(index)
print(df3)