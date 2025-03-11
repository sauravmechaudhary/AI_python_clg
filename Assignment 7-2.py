print("Problem 2: Decision tree")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Read data
data_suv = pd.read_csv("suv.csv")

# Pick Age and Estimated Salary as the features and Purchased as the target variable.
X_suv = data_suv[["Age", "EstimatedSalary"]]
y_suv = data_suv["Purchased"]

# Split data 80/20 %
X_train_suv, X_test_suv, y_train_suv, y_test_suv = train_test_split(X_suv, y_suv, test_size=0.2, random_state=20)

# scaling using standard scaler
scaler = StandardScaler()
X_train_suv = scaler.fit_transform(X_train_suv)
X_test_suv = scaler.transform(X_test_suv)

# Train a decision tree classifier with entropy criterion and predict on test set.
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train_suv, y_train_suv)
y_pred_entropy = dt_entropy.predict(X_test_suv)

# output with entropy criterion
print("Decision Tree with Entropy Criterion:")
print(confusion_matrix(y_test_suv, y_pred_entropy))
print(classification_report(y_test_suv, y_pred_entropy))

# Decision tree with gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train_suv, y_train_suv)
y_pred_gini = dt_gini.predict(X_test_suv)

# output with gini
print("Decision Tree with Gini Criterion:")
print(confusion_matrix(y_test_suv, y_pred_gini))
print(classification_report(y_test_suv, y_pred_gini))


print("The Gini criterion is faster and creates simpler trees, making it efficient for large datasets, while the Entropy criterion provides more detailed splits but can be slower and lead to deeper trees. Both methods usually give similar accuracy, so the best choice depends on the dataset and model performance.")