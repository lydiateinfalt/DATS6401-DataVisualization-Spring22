import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


data_1D = pd.Series([1,2,3,4, np.NAN])
print(data_1D)
np.random.seed(123)
index = np.arange(1,7)
data_2D = pd.DataFrame(data = np.random.randn(6,4), index = index, columns=list('ABCD'))
print(data_2D)
data_2D.info()

df = pd.DataFrame({'Gender': ['female','female', 'male', 'male', 'male'],
                    'Age' : [25, 18, '', 52, 33],
                    'Weight': [250, 280, np.nan, 210, 330],
                   'Location': ['CA', 'DC', 'VA', 'MA', 'VA'],
                   'Arrest Record' : ['No', 'Yes', 'Yes', 'No', np.nan]
                    })

print(df)

name = sn.get_dataset_names()
print(name)

df =sn.load_dataset('car_crashes')
col_name = df.columns
print(df.head(10))
print(df.describe())
print(df.columns)
total = df[col_name[0]].values
plt.figure()
#plt.plot(total)
df[:10].plot(y='total')
plt.show()


#df2 = df[['speeding', 'alcohol', 'ins_premium', 'total']]
df2 = df[['speeding', 'alcohol', 'total']]
plt.figure()
df2.plot()
plt.show()

df2_z_score = (df2 - df2.mean())/df2.std()
plt.figure()
df2_z_score[30:].plot()
plt.grid()
plt.show()

speeding = df2.speeding
alcohol = df2.alcohol
total = df2.total


def corr_coeff(x, y):
    diff_x = []
    diff_y = []
    r = []
    diff_x_squared = []
    diff_y_squared = []

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    for i in range(len(x)):
        x_hat = (x[i] - mean_x)
        diff_x.append(x_hat)
        diff_x_squared.append(np.square(x_hat))
        y_hat = (y[i]- mean_y)
        r.append((x_hat * y_hat))
        diff_y_squared.append(np.square(y_hat))
    return (np.sum(r))/(np.sqrt(np.sum(diff_x_squared))*np.sqrt(np.sum(diff_y_squared)))

r_alc = corr_coeff(total, alcohol)
r_speed = corr_coeff(total, speeding)

plt.scatter(alcohol, total)
plt.title(f'correlation coefficient {r_alc: .2f}')
plt.grid()
plt.xlabel('Alcohol')
plt.ylabel('Total car crashes')
plt.show()

plt.figure()
plt.scatter(speeding, total)
plt.grid()
plt.xlabel('Speeding')
plt.ylabel('Total car crashes')
plt.show()

