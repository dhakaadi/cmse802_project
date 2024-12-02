# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed for convergence
model.fit(X_train, Y_train)

# Make predictions and evaluate the model
Y_pred = model.predict(X_test)
print("Logistic Regression Accuracy Score:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Decision Tree model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=7)
dt_model.fit(X_train, Y_train)
y_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy Score:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

# Random Forest model
rf_model = RandomForestClassifier(random_state=7)
rf_model.fit(X_train, Y_train)
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy Score:", accuracy_score(Y_test, rf_y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(Y_test, rf_y_pred))
print("Random Forest Classification Report:\n", classification_report(Y_test, rf_y_pred))

# Plotting confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(Y_test, rf_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Random Forest Confusion Matrix')
plt.show()

# K-Nearest Neighbors (k-NN) model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_y_pred = knn_model.predict(X_test)
print("K-Nearest Neighbors Accuracy Score:", accuracy_score(Y_test, knn_y_pred))
print("K-Nearest Neighbors Confusion Matrix:\n", confusion_matrix(Y_test, knn_y_pred))
print("K-Nearest Neighbors Classification Report:\n", classification_report(Y_test, knn_y_pred))

# Support Vector Machine (SVM) model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model = SVC(random_state=7)
svm_model.fit(X_train, Y_train)
svm_y_pred = svm_model.predict(X_test)
print("Support Vector Machine Accuracy Score:", accuracy_score(Y_test, svm_y_pred))
print("Support Vector Machine Confusion Matrix:\n", confusion_matrix(Y_test, svm_y_pred))
print("Support Vector Machine Classification Report:\n", classification_report(Y_test, svm_y_pred, zero_division=1))

# Gaussian Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)
gnb_y_pred = gnb_model.predict(X_test)
print("Gaussian Naive Bayes Accuracy Score:", accuracy_score(Y_test, gnb_y_pred))
print("Gaussian Naive Bayes Confusion Matrix:\n", confusion_matrix(Y_test, gnb_y_pred))
print("Gaussian Naive Bayes Classification Report:\n", classification_report(Y_test, gnb_y_pred))

# Multi-Layer Perceptron (MLP) model
mlp_model = MLPClassifier(max_iter=1000, random_state=7)
mlp_model.fit(X_train, Y_train)
mlp_y_pred = mlp_model.predict(X_test)
print("Multi-Layer Perceptron Accuracy Score:", accuracy_score(Y_test, mlp_y_pred))
print("Multi-Layer Perceptron Confusion Matrix:\n", confusion_matrix(Y_test, mlp_y_pred))
print("Multi-Layer Perceptron Classification Report:\n", classification_report(Y_test, mlp_y_pred))

# XGBoost model
xgb_model = XGBClassifier(random_state=7)
xgb_model.fit(X_train, Y_train)
xgb_y_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy Score:", accuracy_score(Y_test, xgb_y_pred))
print("XGBoost Confusion Matrix:\n", confusion_matrix(Y_test, xgb_y_pred))
print("XGBoost Classification Report:\n", classification_report(Y_test, xgb_y_pred))
