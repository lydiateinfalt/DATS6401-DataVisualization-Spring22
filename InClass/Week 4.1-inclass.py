import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')
#plt.style.use('seaborn-deep')
x = np.linspace(0, 2*np.pi,1000)
y = np.sin(x)
y2 = np.cos(x)
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

plt.figure(figsize=(12,8))
plt.plot(x,y,lw=4, label = 'sin(x)', color = 'r', marker = 'H', ms=20)
plt.plot(x,y2,lw=4, label = 'cos(x)', color = 'c', marker = 'D', ms=20)
plt.legend(fontsize = 20, loc = 'center left')
plt.title("Sin of x", fontdict=font2)
plt.grid(axis = 'y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Samples', fontdict=font1)
plt.xlabel('Magnitude', fontdict=font1)
plt.show()