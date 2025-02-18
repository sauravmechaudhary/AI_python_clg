import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

matplotlib.use('TkAgg')


file_path = "Auto.csv"  # Ensure correct file path
df = pd.read_csv(file_path)

print(df.head())

print(df.info())

df = df.drop(columns=['name', 'origin'])  # Drop categorical columns
y = df['mpg']
X = df.drop(columns=['mpg'])

df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-3, 2, 50)  # Logarithmic scale from 0.001 to 100

ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test_scaled)))

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test_scaled)))

plt.figure(figsize=(10, 5))
plt.plot(alphas, ridge_scores, label='Ridge R²', marker='o')
plt.plot(alphas, lasso_scores, label='Lasso R²', marker='s')
plt.xscale("log")  # Log scale for better visualization
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² Score")
plt.title("R² Score vs Alpha for Ridge and Lasso Regression")
plt.legend()
plt.show()

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]
best_ridge_score = max(ridge_scores)
best_lasso_score = max(lasso_scores)

print(f"Best Ridge Alpha: {best_alpha_ridge}, Best R² Score: {best_ridge_score:.4f}")
print(f"Best Lasso Alpha: {best_alpha_lasso}, Best R² Score: {best_lasso_score:.4f}")

"""
Findings:
1. The dataset was loaded, and 'name' and 'origin' columns were removed as they are non-numeric.
2. The 'mpg' variable was set as the target (y), and all other numerical features were used as predictors (X).
3. Data was split into 80% training and 20% testing.
4. Ridge and Lasso regression were applied with multiple alpha values (0.001 to 100).
5. The best alpha values were found by maximizing the R² score on the test set.
6. R² scores were plotted against alpha for visualization.
7. The best Ridge and Lasso alpha values indicate the optimal level of regularization for the dataset.
"""
