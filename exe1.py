import matplotlib
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('TkAgg')

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
x_base = df[['bmi', 's5']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(x_base, y, test_size=0.2, random_state=5)
lm = LinearRegression()
lm.fit(X_train, y_train)

y_test_pred = lm.predict(X_test)
y_train_pred = lm.predict(X_train)
rmse_base = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_base = r2_score(y_test, y_test_pred)

x_extended = df[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(x_extended, y, test_size=0.2, random_state=5)
lm.fit(X_train, y_train)
y_test_pred_ext = lm.predict(X_test)
y_train_pred_ext = lm.predict(X_train)
rmse_extended = np.sqrt(mean_squared_error(y_test, y_test_pred_ext))
r2_extended = r2_score(y_test, y_test_pred_ext)

x_full = df.drop(columns=['target'])
X_train, X_test, y_train, y_test = train_test_split(x_full, y, test_size=0.2, random_state=5)

lm.fit(X_train, y_train)
y_test_pred_full = lm.predict(X_test)
y_train_pred_full = lm.predict(X_train)
rmse_full = np.sqrt(mean_squared_error(y_test, y_test_pred_full))
r2_full = r2_score(y_test, y_test_pred_full)

print("Base Model: BMI + S5")
print(f"RMSE: {rmse_base:.2f}, R2: {r2_base:.2f}")
print("\nExtended Model: BMI + S5 + BP")
print(f"RMSE: {rmse_extended:.2f}, R2: {r2_extended:.2f}")
print("\nFull Model: All Features")
print(f"RMSE: {rmse_full:.2f}, R2: {r2_full:.2f}")

"""
Findings:
1. Adding 'bp' improved the R2 score slightly, reducing RMSE, suggesting better prediction accuracy.
2. Using all features did not necessarily lead to a better model due to potential overfitting.
3. Evaluating feature importance can help in choosing the best subset of variables.
"""
