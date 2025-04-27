import matplotlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


filePath = 'weight-height.csv'
data = pd.read_csv(filePath)


lengths_in_inches = data['Height'].values
weights_in_pounds = data['Weight'].values

lengths_in_cm = lengths_in_inches * 2.54
weights_in_kg = weights_in_pounds * 0.453592

mean_length_cm = np.mean(lengths_in_cm)
mean_weight_kg = np.mean(weights_in_kg)

plt.figure(figsize=(8, 6))
plt.hist(lengths_in_cm, bins=20, color='Green', edgecolor='Blue', alpha=0.9)
plt.title('Histogram of Heights (cm)', fontsize=18)
plt.xlabel('Height (cm)', fontsize=20)
plt.ylabel('Frequency', fontsize=22)
plt.grid(axis='y', linestyle='--', alpha=0.9)

plt.show()

print(f"Mean Height: {mean_length_cm:.2f} cm")
print(f"Mean Weight: {mean_weight_kg:.2f}kg")