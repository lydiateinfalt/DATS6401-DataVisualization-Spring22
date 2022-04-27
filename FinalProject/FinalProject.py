import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

os.chdir("C:\\Users\\Teinfalt\\Documents\\GitHub\\DATS6401-DataVisualization-Spring22\\FinalProject")
data = pd.read_csv("secondary_data_generated.csv", sep=';', header=0)
columns = data.columns
a = data.isnull().sum()/len(data)
print(a)
variable = []
df = data.copy()
for i in range(0, len(columns)):
    if a[i] >= .2:
        variable.append((columns[i]))

df.drop(variable, axis = 1, inplace=True)
print(df)
print(df.head())
print(df.describe())
print(df.info())
df1 = df.copy()
cols = df1.columns
#class
dict_class = {'p': 'poisonous', 'e': 'edible'}
#cap shape dictionary
dict_cs = {'b' : 'bell', 'c' : 'conical', 'x' : 'convex', 'f' : 'flat', 's': 'sunken', 'p' : 'spherical' , 'o' : 'others'}
dict_capcolor = {'n': 'brown', 'b': 'buff', 'g':'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l':'blue','o' :'orange', 'k':'black'}
dict_ga = {'a': 'adnate', 'x': 'adnexed', 'd' :'decurrent', 'e': 'free','s' :'sinuate', 'p': 'pores', 'f': 'none', '?': 'unknown'}
dict_stemroot = {'b': 'bulbous', 's': 'swollen', 'c': 'club', 'u': 'cup', 'e':'equal','z': 'rhizomorphs', 'r': 'rooted'}
dict_gillcolor = dict_capcolor.copy()
dict_gillcolor['f'] = 'none'
dict_ringtype = { 'c':'cobwebby', 'e': 'evanescent', 'r': 'flaring', 'g': 'grooved', 'l': 'large', 'p': 'pendant', 's': 'sheathing', 'z': 'zone', 'y': 'scaly', 'm': 'movable', 'f': 'none', '?':'unknown'}
dict_habitat = {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'h': 'heaths','u': 'urban', 'w': 'waste', 'd': 'woods'}
dict_season = {'s': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'}
df1['class'] = df1['class'].map(dict_class)
df1['cap-shape']= df1['cap-shape'].map(dict_cs)
df1['cap-color'] =df1['cap-color'].map(dict_capcolor)
df1['does-bruise-or-bleed']=df1['does-bruise-or-bleed'].map({'t': 'True', 'f': 'False'})
df1['gill-attachment'] = df1['gill-attachment'].map(dict_ga)
df1['gill-color'] = df1['gill-color'].map(dict_gillcolor)
df1['stem-color'] = df1['stem-color'].map(dict_gillcolor)
df1['has-ring'] = df1['has-ring'].map({'t': 'True', 'f': 'False'})
df1['ring-type']=df1['ring-type'].map(dict_ringtype)
df1['habitat']=df1['habitat'].map(dict_habitat)
df1['season']=df1['season'].map(dict_season)
df1.to_csv("mushrooms.csv")
print("")


plt.grid(True)


# for i in range(len(cols)):
#     sns.histplot(df1, x=cols[i])
#     plt.show()

mushroom = df1.copy()

sns.histplot(data = mushroom, x = 'cap-diameter', bins=30)
plt.title('Histogram of Mushroom Cap-Diameter')
plt.show()

sns.histplot(data = mushroom,x = 'stem-height', bins=30)
plt.title('Histogram of Mushroom Stem-Height')
plt.show()

sns.histplot(data = mushroom,x = 'stem-width',bins=30)
plt.title('Histogram of Mushroom Stem-Width')
plt.show()

sns.histplot(mushroom, x = "cap-diameter", hue = "class", element = "step", bins = 15)
plt.title('Histogram of Mushroom Cap-Diameter')
plt.show()

sns.histplot(data = mushroom,x = 'stem-height',hue = 'class', multiple = 'stack', bins = 15)
plt.title('Histogram of Mushroom Stem-Height')
plt.show()

sns.displot(mushroom, x = 'stem-width',hue = 'class', multiple = 'dodge', bins = 30)
plt.title('Histogram of Mushroom Stem-Width')
plt.show()

sns.displot(data = mushroom,x = 'cap-diameter',hue= 'class', bins = 15)
plt.title('Distribution Plot of Mushroom Cap-Diameter')
plt.show()

sns.displot(data = mushroom, x="cap-diameter", hue= 'class', stat = 'density', bins = 15)
plt.title('Distribution Plot of Mushroom Cap-Diameter')
plt.show()

sns.displot(data = mushroom, x="stem-height", hue= 'class', stat = 'density', bins = 15)
plt.title('Distribution Plot of Mushroom Stem-Height')
plt.show()

sns.displot(data = mushroom, x="stem-width", hue= 'class', stat = 'density', bins = 30)
plt.title('Distribution Plot of Mushroom Stem-Width')
plt.show()

sns.countplot(data = mushroom, x = 'cap-color', hue = 'class')
plt.title('Count Plot of Mushroom Cap-Color')
plt.show()

sns.displot(mushroom, x = 'stem-height', hue = 'class', kind='kde', multiple = 'stack')
plt.title('Distribution Plot of Mushroom Stem-Height')
plt.show()

sns.displot(mushroom, x = 'stem-height', hue = 'class', kind='kde', fill = True)
plt.title('Distribution Plot of Mushroom Stem-Height')
plt.show()

poisonous = mushroom['class'].value_counts()['poisonous']
edible = mushroom['class'].value_counts()['edible']

class_count = np.array([poisonous, edible])
fig, ax = plt.subplots(figsize=(8,8))
ax.pie(class_count, labels=['Poisonous', 'Edible'], autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Poisonous versus Edible Mushrooms")
plt.show()

p = mushroom[mushroom['class'] == 'poisonous']
e = mushroom[mushroom['class'] == 'edible']

cap_color = p['cap-color'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
ax.pie(cap_color, labels=dict_capcolor, autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Mushroom Cap Colors")
plt.legend(dict_capcolor.values(), loc = 'best')
plt.tight_layout()
plt.show()

cap_shape = mushroom['cap-shape'].value_counts()
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(cap_shape, labels=dict_cs.values(), autopct='%.1f%%')
ax.axis('square')
ax.set_title("Pie chart of Mushroom Cap Shape")
plt.legend(dict_cs.values(), loc = 'best')
plt.tight_layout
plt.show()

sns.scatterplot(data = mushroom, x = 'stem-width', y = 'stem-height', hue = 'class')
plt.title('Scatter plot of Mushroom Stem-Height vs. Stem-Width')
plt.show()

sns.displot(data=mushroom, x="cap-diameter", hue="cap-color", col="class", bins =15)
plt.show()

sns.displot(data=mushroom, x="cap-diameter", hue="cap-shape", col="class", bins =15)
plt.show()

sns.displot(data=mushroom, x="cap-diameter", hue="stem-color", col="class", bins =15)
plt.show()

sns.displot(data=mushroom, x="class", hue="habitat", col="season", bins =15)
plt.show()

sns.kdeplot(data = mushroom, x="stem-width", y="stem-height", hue= 'class', fill = True)
plt.show()
sns.kdeplot(data = mushroom, x="stem-height", y="cap-diameter", hue= 'class', fill = True)
plt.show()
sns.kdeplot(data = mushroom, x="stem-width", y="cap-diameter", hue= 'class', fill = True)
plt.show()

sns.catplot(x="season", hue = "habitat", col = "class", data = mushroom, kind = "count",height=4, aspect=.7)
plt.show()

sns.lineplot(x='cap-shape', y ='stem-height', data=mushroom, hue='class')
plt.show()

sns.lineplot(x='cap-shape', y ='stem-height', data=mushroom, hue='cap-color')
plt.show()

sns.lineplot(x='cap-color', y ='stem-height', data=mushroom, hue='class')
plt.show()


sns.lineplot(x='gill-color', y ='stem-height', data=mushroom, hue='class')
plt.show()

