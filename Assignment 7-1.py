print("Problem 1: SVM")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# 0> Read the dataset
data = pd.read_csv("data_banknote_authentication.csv")

# 1> Pick the column named "class" as target variable y and all other columns as feature variables X.
X = data.drop(columns=["class"])
y = data["class"]

# 2> Split the data into training and testing sets with 80/20 ratio and random_state=20.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# 4> Predict on the testing data and compute the confusion matrix and classification report
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# output for for liner kernel
print("SVM with Linear Kernel:")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# for RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# Output for RBF kernel
print("SVM with RBF Kernel:")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

#In my words

print("The linear SVM works best when the data can be separated with a straight line, making it faster. The RBF SVM is better for complex data with curved patterns but takes more time to train.")