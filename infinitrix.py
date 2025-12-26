
# Infinitrix.ipynb

# Original file is located at
#     https://colab.research.google.com/drive/15lYNABv-J0hOwspe93CFo4mm6PT4pGpA

# A LOGISTIC REGTRESSION IRIS CLASSIFIER MODEL
# 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from google.colab import files
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing all the Neccessary Libraries for data extraction and data preprocessing

files.upload()

# Uploadiung the file from the local storage to the Remote server

df = pd.read_csv("iris (2).csv")
print(df.head())

# Checking the required dataframe to be compatible for training or not




feature = df[['sepal_length' , 'sepal_width' , 'petal_length' , 'petal_width']]
target = df[['species']]

# Dividing the data into features and target variables

target = target.replace({'setosa': 1, 'versicolor': 2, 'virginica': 3})

# Renaming the label classes to make them code relevant 

model =LogisticRegression()

# Declaring the LOGISTIC REGRESSION model 

X_train , X_test ,y_train ,y_test = train_test_split(feature , target , test_size=0.2 , shuffle =True ,random_state =42)

#  Performing Train Test Split for training and testing purposes

model.fit(X_train , y_train)

# fitting data into LOGISTIC REGRESSION MODEL

y_pred = model.predict(X_test)

# Forward pass of test dataset

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1", "Class 2"]
)

disp.plot(cmap="Blues")
plt.show()

t = 0
for i in range(0, y_pred.size):
  if y_pred[i] == y_test.iloc[i, 0]:
    t += 1
print(t / y_pred.size)

#  Checking the credibility of the model by checking number of correct and incorrect predicts

pred = model.predict([[	4.9 ,2.0,3.0, 1.0]])

if(pred == 1):
  print('setosa')
elif(pred == 2):
  print('versicolor')
else:
  print('virginica')

# PRINTING THE FINAL ANSWER 

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------


#LINEAR REGRESSION MODEL

model =LinearRegression()

# Declaring the LOGISTIC REGRESSION model 

X_train , X_test ,y_train ,y_test = train_test_split(feature , target , test_size=0.2 , shuffle =True ,random_state =42)

#  Performing Train Test Split for training and testing purposes

model.fit(X_train , y_train)

# fitting data into LOGISTIC REGRESSION MODEL

y_pred = model.predict(X_test)
#converting y_pred logits to actual class labels
y_pred = np.argmax(y_pred, axis=1)

# Forward pass of test dataset

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1", "Class 2"]
)

disp.plot(cmap="Blues")
plt.show()

t = 0
for i in range(0, y_pred.size):
  if y_pred[i] == y_test.iloc[i, 0]:
    t += 1
print(t / y_pred.size)

#  Checking the credibility of the model by checking number of correct and incorrect predicts

pred = model.predict([[	4.9 ,2.0,3.0, 1.0]])

if(pred == 1):
  print('setosa')
elif(pred == 2):
  print('versicolor')
else:
  print('virginica')

# PRINTING THE FINAL ANSWER 