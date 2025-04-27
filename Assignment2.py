import matplotlib
import numpy as np

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

x = np.linspace(-10, 10, 400)

y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

plt.plot(x, y1, linestyle='-', color='Red', label='y = 2x + 1')
plt.plot(x, y2, linestyle='--', color='green', label='y = 2x + 2')
plt.plot(x, y3, linestyle='-.', color='purple', label='y = 2x + 3')

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Graph of y = 2x + c for c = 1, 2, 3')

plt.grid(True)
plt.legend()
plt.show()
