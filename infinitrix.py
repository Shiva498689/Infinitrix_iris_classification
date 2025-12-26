"""
Infinitrix.ipynb

Description:
    Iris Classification utilizing Logistic Regression and Linear Regression.
    This script performs data loading, preprocessing, model training, 
    and evaluation using the Scikit-Learn framework on Google Colab.

Author: Shiva Dubey
"""

# ==========================================
# 1. Setup and Data Preprocessing
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Uploading the file from local storage to the Remote Colab server
files.upload()

# Loading the dataset
df = pd.read_csv("iris (2).csv")
print("Dataset Head:\n", df.head())

# Feature Selection
feature = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = df[['species']]

# Label Encoding: Converting string labels to numerical values
target = target.replace({'setosa': 1, 'versicolor': 2, 'virginica': 3})

# ==========================================
# 2. Model I: Logistic Regression
# ==========================================

print("\n--- Training Logistic Regression Model ---")

model_log = LogisticRegression()

# Splitting data into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, shuffle=True, random_state=42
)

# Training the model
model_log.fit(X_train, y_train)

# Forward pass (Prediction) on test data
y_pred_log = model_log.predict(X_test)

# Evaluation: Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix (Logistic):\n", cm_log)

# Visualization: Confusion Matrix
disp_log = ConfusionMatrixDisplay(
    confusion_matrix=cm_log,
    display_labels=["Class 1", "Class 2", "Class 3"]
)
disp_log.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Accuracy Check (Manual Calculation)
correct_preds = 0
for i in range(0, y_pred_log.size):
    if y_pred_log[i] == y_test.iloc[i, 0]:
        correct_preds += 1
print(f"Logistic Model Accuracy: {correct_preds / y_pred_log.size}")

# Single Instance Prediction
single_pred_log = model_log.predict([[4.9, 2.0, 3.0, 1.0]])

print("Prediction for [4.9, 2.0, 3.0, 1.0]: ", end="")
if single_pred_log == 1:
    print('Setosa')
elif single_pred_log == 2:
    print('Versicolor')
else:
    print('Virginica')


# ==========================================
# 3. Model II: Linear Regression
# ==========================================

print("\n--- Training Linear Regression Model ---")

model_lin = LinearRegression()

# Re-splitting data (Standard practice to ensure clean state)
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, shuffle=True, random_state=42
)

# Training the model
model_lin.fit(X_train, y_train)

# Forward pass (Prediction) on test data
y_pred_lin = model_lin.predict(X_test)

# Converting logits/continuous outputs to class labels
# Note: Using argmax as per original logic
y_pred_lin = np.argmax(y_pred_lin, axis=1)

# Evaluation: Confusion Matrix
cm_lin = confusion_matrix(y_test, y_pred_lin)
print("Confusion Matrix (Linear):\n", cm_lin)

# Visualization: Confusion Matrix
disp_lin = ConfusionMatrixDisplay(
    confusion_matrix=cm_lin,
    display_labels=["Class 1", "Class 2", "Class 3"]
)
disp_lin.plot(cmap="Blues")
plt.title("Linear Regression Confusion Matrix")
plt.show()

# Accuracy Check (Manual Calculation)
correct_preds_lin = 0
for i in range(0, y_pred_lin.size):
    if y_pred_lin[i] == y_test.iloc[i, 0]:
        correct_preds_lin += 1
print(f"Linear Model Accuracy: {correct_preds_lin / y_pred_lin.size}")

# Single Instance Prediction
single_pred_lin = model_lin.predict([[4.9, 2.0, 3.0, 1.0]])

# Note: Linear regression prediction logic
print("Prediction for [4.9, 2.0, 3.0, 1.0]: ", end="")
if single_pred_lin == 1:
    print('Setosa')
elif single_pred_lin == 2:
    print('Versicolor')
else:
    print('Virginica')