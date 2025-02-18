import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('TkAgg')

file_path = "50_Startups.csv"
df = pd.read_csv(file_path, delimiter=",")

print(df.head())

print(df.info())

df_numeric = df.drop(columns=['State'])

plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

df = pd.get_dummies(df, columns=['State'], drop_first=True)

selected_features = ['R&D Spend', 'Marketing Spend']
y = df['Profit']
X = df[selected_features]

plt.figure(figsize=(12, 5))
for i, feature in enumerate(selected_features):
    plt.subplot(1, 2, i + 1)
    plt.scatter(df[feature], df['Profit'])
    plt.xlabel(feature)
    plt.ylabel('Profit')
    plt.title(f"Profit vs {feature}")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"RMSE (train): {rmse_train:.2f}, R2 (train): {r2_train:.2f}")
print(f"RMSE (test): {rmse_test:.2f}, R2 (test): {r2_test:.2f}")

"""
Findings:
1. Profit along with expenditures in R&D, marketing, and administration.
2. Correlation analysis suggests 'R&D Spend' and 'Marketing Spend' have the highest correlation with profit.
3. Scatter plots confirm a nearly linear relationship between these variables and profit.
4. The 'State' column was one-hot encoded but not used in the model since it has little impact.
5. The linear regression model was trained using an 80/20 split.
6. RMSE and R2 values indicate model performance.
"""
