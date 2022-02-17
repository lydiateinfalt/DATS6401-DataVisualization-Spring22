import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot

fig, ax = plt.subplots(1,1)
explode = (.03, .03, .3, .03, .03)
label = ['C', 'C++', 'Java', 'Python', 'PHP']
score_men = [23, 17, 35, 29, 12]
score_women = [35, 25,10, 15,30]

score = [23, 17, 35, 29, 12]
ax.pie(score, labels = label, explode = explode, autopct='%1.3f%%')
ax.axis('square')
plt.show()
print(100*(23/np.sum(score)))

#Bar plots
plt.figure()
plt.bar(label, score)
plt.xlabel('Language')
plt.ylabel('Score')
plt.title("Score vs. Programming Languages")
plt.show()

#Stacked barplot
plt.bar(label, score_men, label= 'Men')
plt.bar(label, score_women, label= 'Women', bottom=score_men)
plt.xlabel('Language')
plt.legend()
plt.ylabel('Score')
plt.title("Simple Stack Bar Plot")
plt.show()

#Group bat plot
plt.figure()
width = 0.4
x = np.arange(len(label))
ax.bar(x-width/2, score_men, width, label = 'Men')
ax.bar(x+width/2, score_women, width, label = 'Women')
ax.set_xlabel('Language')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.set_ylabel('Score')
ax.set_title('Simple Stack Bar Plot')
ax.legend()
plt.show()

#Horizontal stacked bar
fig, ax = plt.subplots()
width = 0.4
x = np.arange(len(label))
ax.barh(x-width/2, score_men, width, label = 'Men')
ax.set_ylabel('Language')
ax.set_xlabel('Score')
ax.set_yticks(x)
ax.set_yticklabels(label)
ax.set_title('Simple Horizontal Bar Plot')
ax.legend()
plt.show()

np.random.seed(10)
data1 = np.random.normal(100, 10,1000 )
data2 = np.random.normal(90, 20,1000 )
data3 = np.random.normal(80, 30,1000 )
data4 = np.random.normal(70, 40,1000 )
data = [data1, data2, data3, data4]
plt.boxplot(data)
plt.xticks([1,2,3,4])
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('Box plot')
plt.show()

#histogram
plt.hist(data, bins=50)
plt.show()
#if data does not normal distribution, you cannot use parametric methods like t-test

#qq plot to check normality
data1 = np.random.normal(0,1,1000)
plt.figure()
qqplot(data1, line='45')
plt.title('QQ-plot')
plt.show()

plt.figure()
plt.hist(data1,bins=50)
plt.show()

x = range(1,20)
#y = [1,4,6,8,4]
y = np.random.normal(10,2,len(x))
plt.plot(x,y,color='blue', lw=3)
plt.fill_between(x,y, label = 'Area')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc= 'upper left')
plt.title("Simple Area Plot")
plt.grid()
plt.show()

x = np.linspace(0, 2*np.pi, 41)
y = np.exp(np.sin(x))
#(markers, stemlines, baseline) = plt.stem()