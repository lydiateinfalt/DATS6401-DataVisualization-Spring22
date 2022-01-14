import numpy as np
import matplotlib.pyplot as plt


N = 1000
mean_x = 0
mean_y = 2
std_x = np.sqrt(1)
std_y = np.sqrt(5)
#Random number with normal distribution
x = np.random.normal(mean_x, std_x, N)
y = np.random.normal(mean_y, std_y, N)

# line plot
plt.figure(figsize=(12,8))
plt.plot(x, 'g', label = 'x')
plt.plot(y, 'r', label = 'y')
plt.legend()
plt.grid()
plt.show()


#histogram
plt.figure(figsize=(12,8))
plt.hist(x, bins=50, label = 'x', alpha = .5)
plt.hist(y, bins=50, label = 'y', alpha = .5)
plt.legend()
plt.grid()
plt.show()

#scatterplot
plt.figure(figsize=(12,8))
plt.scatter(x, y)
plt.legend()
plt.grid()
plt.show()

