import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style='darkgrid')

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
diamonds = sns.load_dataset('diamonds')
penguins = sns.load_dataset('penguins')
titanic = sns.load_dataset('titanic')

print(flights.describe())
sns.lineplot(data = flights,
             x = 'year',
             y = 'passengers',
             hue = 'month')
plt.show()

data = np.random.normal(size=(20,6)) + np.arange(6)/2
sns.boxplot(data = data)
plt.show()



sns.regplot(data = tips,
           x = 'total_bill',
           y = 'tip')
plt.show()

sns.boxplot(data = tips[['total_bill', 'tip']])
plt.show()

sns.relplot(data = tips,
           x = 'total_bill',
           y = 'tip',
            hue = 'sex')
plt.show()

sns.relplot(data = tips,
           x = 'total_bill',
           y = 'tip',
            hue = 'day')
plt.show()

#relational plots
sns.relplot(data = flights,
           x = 'year',
           y = 'passengers',
            kind = 'line')
plt.show()

sns.relplot(data = flights,
           x = 'year',
           y = 'passengers',
            kind = 'line',
            hue = 'month')
plt.show()


sns.relplot(data = tips,
           x = 'total_bill',
           y = 'tip',
            kind = 'scatter',
            hue = 'day',
            col = 'time')
plt.show()

sns.relplot(data = tips,
           x = 'total_bill',
           y = 'tip',
            kind = 'scatter',
            hue = 'day',
            col = 'smoker')
plt.show()

sns.relplot(data = tips,
           x = 'total_bill',
           y = 'tip',
            kind = 'scatter',
            hue = 'day',
            col = 'time',
            row = 'smoker')
plt.show()

df = flights.pivot('month', 'year', 'passengers')
sns.heatmap(df,
            center = df.loc['Jul',1960],
            cmap = "YlGnBu")
plt.show()

sns.countplot(data = tips,
              x = 'day')
plt.show()

sns.countplot(data = tips,
              x = 'day',
              order = tips['day'].value_counts().index)
plt.show()

sns.countplot(data = tips,
              x = 'day',
              order = tips['day'].value_counts().index[::-1])
plt.show()

sns.countplot(data = tips,
              x = 'day',
              order = tips['day'].value_counts(ascending=False).index)
plt.show()

sns.countplot(data = diamonds,
              y = 'clarity',
              order = diamonds['clarity'].value_counts(ascending = False).index)
plt.title('Clarity per sales count')
plt.show()

sns.countplot(data = diamonds,
              y = 'cut',
              order = diamonds['cut'].value_counts(ascending = False).index)
plt.title('Cut per sales count')
plt.show()

sns.countplot(data = diamonds,
              y = 'color',
              order = diamonds['color'].value_counts(ascending = False).index)
plt.title('Color per sales count')
plt.show()

sns.color_palette('Paired')
sns.countplot(data = titanic,
              x = 'class',
              hue = 'who',
              palette = ['#432371', '#FAAE7B', '#AAAE7B'])
plt.show()
sns.countplot(data = titanic,
              x = 'class',
              hue = 'who',
              palette = ['#432371', '#FAAE7B', '#AAAE7B'])
plt.show()


sns.countplot(data = titanic,
              y = 'class',
              hue = 'who',
              palette = ['#432371', '#FAAE7B', '#AAAE7B'])
plt.show()

sns.pairplot(data = penguins,
             hue = 'sex')
plt.show()

sns.countplot(data = tips,
              x = 'sex',
              hue = 'smoker')
plt.show()

sns.kdeplot(data = tips,
            x = 'total_bill',
            bw_adjust = 0.2)
plt.show()

sns.kdeplot(data = tips,
            x = 'total_bill',
            hue = 'time')
plt.show()

sns.kdeplot(data = tips,
            x = 'total_bill',
            hue = 'time',
            multiple = 'fill')
plt.show()

sns.kdeplot(data = diamonds,
            x = 'price',
            log_scale = False)
plt.show()

sns.kdeplot(data = diamonds,
            x = 'price',
            log_scale = True,
            kind = 'scatter',
            hue = 'clarity',
            palette = 'crest',
            alpha = 0.5,
            linewidth = 0,
            fill = True)
plt.show()

sns.kdeplot(data = tips,
            x='total_bill',
            y = 'tip')
plt.show()