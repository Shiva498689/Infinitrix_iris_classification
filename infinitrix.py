"""
Infinitrix.ipynb

Description:
    Iris Classification utilizing Logistic Regression and Linear Regression.
    Includes robust Data Preprocessing (Scaling, Cleaning) and Evaluation.

Author: Shiva Dubey
"""

# ==========================================
# 1. Imports and Setup
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for prettier plots (optional but impressive)
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Uploading the file from local storage to the Remote Colab server
files.upload()

# Loading the dataset
df = pd.read_csv("iris.csv")

# ==========================================
# 2. Advanced Data Preprocessing
# ==========================================
print("\n--- Data Preprocessing & Cleaning ---")

# 2.1 Sanity Check: Handling Missing Values
if df.isnull().sum().sum() > 0:
    print(f"Detected {df.isnull().sum().sum()} missing values. Filling with mean...")
    df.fillna(df.mean(), inplace=True)
else:
    print("Dataset is clean: No missing values detected.")

# 2.2 Sanity Check: Removing Duplicates
initial_len = len(df)
df = df.drop_duplicates()
print(f"Duplicate check: Removed {initial_len - len(df)} duplicate rows.")

# 2.3 Feature Separation
feature = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = df[['species']]

# 2.4 Label Encoding
# Replacing manual dictionary with automatic coding if needed,
# but keeping your logic for specific mapping 1,2,3 for consistency.
target = target.replace({'setosa': 1, 'versicolor': 2, 'virginica': 3})

# 2.5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, shuffle=True, random_state=42
)

# 2.6 Feature Scaling (The "Impressive" Math Step)
# Standardizing features to have Mean=0 and Variance=1 for better optimizer convergence.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data
X_test = scaler.transform(X_test)        # Transform test data using train stats

print("Data successfully preprocessed and scaled.")

# ==========================================
# 3. Model I: Logistic Regression
# ==========================================

print("\n--- Training Logistic Regression Model ---")

model_log = LogisticRegression()
model_log.fit(X_train, y_train.values.ravel()) # .ravel() converts column vector to 1D array

# Forward pass
y_pred_log = model_log.predict(X_test)

# Evaluation
cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix (Logistic):\n", cm_log)

# Visualization
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Setosa", "Versicolor", "Virginica"])
disp_log.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Accuracy Calculation
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Model Accuracy: {accuracy_log:.4f}")

# Single Prediction (Must scale input first!)
sample_input = [[4.9, 2.0, 3.0, 1.0]]
sample_scaled = scaler.transform(sample_input) # Scale the single input
pred_log = model_log.predict(sample_scaled)

print(f"Prediction for {sample_input}: ", end="")
if pred_log == 1: print('Setosa')
elif pred_log == 2: print('Versicolor')
else: print('Virginica')


# ==========================================
# 4. Model II: Linear Regression
# ==========================================

print("\n--- Training Linear Regression Model ---")

model_lin = LinearRegression()
model_lin.fit(X_train, y_train)

# Forward pass
y_pred_lin = model_lin.predict(X_test)

# Converting continuous regression output to nearest class integers
# Rounding is mathematically more appropriate for Linear Regression classification than argmax here
y_pred_lin = np.round(y_pred_lin).astype(int)

# Evaluation
# Note: Linear regression might predict <1 or >3, so we clip values to be safe
y_pred_lin = np.clip(y_pred_lin, 1, 3)

cm_lin = confusion_matrix(y_test, y_pred_lin)
print("Confusion Matrix (Linear):\n", cm_lin)

# Visualization
disp_lin = ConfusionMatrixDisplay(confusion_matrix=cm_lin, display_labels=["Setosa", "Versicolor", "Virginica"])
disp_lin.plot(cmap="Greens")
plt.title("Linear Regression Confusion Matrix")
plt.show()

# Accuracy Calculation
accuracy_lin = accuracy_score(y_test, y_pred_lin)
print(f"Linear Model Accuracy: {accuracy_lin:.4f}")

# Single Prediction
pred_lin = model_lin.predict(sample_scaled) # Use scaled input
pred_lin_rounded = int(np.round(pred_lin))

print(f"Prediction for {sample_input}: ", end="")
if pred_lin_rounded == 1: print('Setosa')
elif pred_lin_rounded == 2: print('Versicolor')
else: print('Virginica')