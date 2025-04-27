import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.scatter(x, y, marker='.', color='blue', label='Data Points')


plt.title('Scatter Plot of Points (x, y)', fontsize=14)
plt.xlabel('x', fontsize=21)
plt.ylabel('y', fontsize=21)


plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()