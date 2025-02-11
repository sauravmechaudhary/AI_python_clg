import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = "weight-height.csv"
data = pd.read_csv(file_path)


X = data[['Height']].values
y = data['Weight'].values


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


plt.scatter(X, y, alpha=0.5, label="Actual Data", color='pink')
plt.plot(X, y_pred, color='yellow', label="Regression Line")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.title("Height vs. Weight Regression")
plt.legend()
plt.show()


rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)


print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
