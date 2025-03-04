# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load the dataset
file_path = "C:/Users/saura/OneDrive - Metropolia Ammattikorkeakoulu Oy/python assignment/Assignment 6/bank.csv"  # Update with correct path
df = pd.read_csv(file_path, delimiter=";")  # Ensure correct delimiter

# Inspect the dataset
print("\nDataset Information:\n")
print(df.info())
print("\nFirst Few Rows:\n")
print(df.head())

# Step 2: Select specific columns for df2
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables into dummy variables
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

# Convert 'y' column to numeric values (0 and 1)
df3['y'] = df3['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 4: Compute and visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Define target variable (y) and explanatory variables (X)
y = df3['y']
X = df3.drop(columns=['y'])

# Step 6: Split the dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Train a Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Step 8: Evaluate the Logistic Regression model
y_pred_log = log_model.predict(X_test)
log_cm = confusion_matrix(y_test, y_pred_log)
log_acc = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Results:")
print("Confusion Matrix:\n", log_cm)
print("Accuracy Score:", log_acc)

# Step 9: Train a K-Nearest Neighbors model (k=3)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Evaluate the KNN model
y_pred_knn = knn_model.predict(X_test)
knn_cm = confusion_matrix(y_test, y_pred_knn)
knn_acc = accuracy_score(y_test, y_pred_knn)

print("\nK-Nearest Neighbors (k=3) Results:")
print("Confusion Matrix:\n", knn_cm)
print("Accuracy Score:", knn_acc)

# Step 10: Compare the results
print("\nComparison of Models:")
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
print(f"K-Nearest Neighbors Accuracy: {knn_acc:.4f}")

if log_acc > knn_acc:
    print("Logistic Regression performed better.")
elif knn_acc > log_acc:
    print("K-Nearest Neighbors performed better.")
else:
    print("Both models performed equally.")

""" 
Step 1 Findings:
- The dataset contains customer data from a bank marketing campaign.
- The target variable is 'y' (whether the customer subscribed to a term deposit).
- Some columns are categorical and need to be converted into numerical format.

Step 3 Findings:
- Categorical columns have been successfully converted into numerical values.
- The 'y' column is now binary (1 for 'yes', 0 for 'no').

Step 4 Findings:
- There is little correlation between most features.
- Some variables have slight positive or negative correlations, but none are extremely strong.

Step 8 Findings:
- Logistic Regression achieved an accuracy score of approximately {:.4f}.
- The confusion matrix shows the number of correct and incorrect predictions.
- Logistic Regression is a simple and efficient model for binary classification.

Step 9 Findings:
- KNN with k=3 achieved an accuracy score of approximately {:.4f}.
- The confusion matrix helps us analyze how many correct and incorrect predictions were made.
- KNN is a distance-based model, so its performance depends on scaling and feature selection.

Step 10 Findings:
- Logistic Regression Accuracy: {:.4f}
- K-Nearest Neighbors Accuracy: {:.4f}
- The better-performing model depends on the dataset, but Logistic Regression is often more stable.
"""