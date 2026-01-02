# Infinitrix AI/ML Domain Induction Task Report

## Iris Species Classification Using Morphological Features

**Author:** Shiva Dubey  
**Roll Number:** 250002069  
**Branch:** Electrical Engineering  
**Institute Email:** ee250002069@iiti.ac.in  

---

## 1. Problem Overview and Motivation

The objective of this project is to design and evaluate supervised machine learning models capable of classifying Iris flower species based on their morphological characteristics. Given a set of continuous-valued features describing the geometry of a flower, the task is to predict its species label among three possible classes: **Setosa, Versicolor, and Virginica**.

This problem is a canonical benchmark in machine learning due to its balanced class distribution, low dimensionality, and partial linear separability. It provides an ideal platform for understanding classification algorithms, optimization techniques, and model limitations.

The motivation of this project is to:
- Understand the mathematical foundations of supervised learning
- Compare appropriate classifiers against naïve baselines
- Build an end-to-end ML pipeline using Scikit-Learn
- Develop critical intuition regarding model selection

---

## 2. Dataset Description and Preprocessing

### 2.1 Dataset Description

The dataset used in this project is derived from the standard **Iris dataset**.

- **Original Samples:** 150
- **Augmented Samples:** ~160
- **Number of Features:** 4
- **Number of Classes:** 3

### Feature Set
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target Variable
- `specie` ∈ {Setosa, Versicolor, Virginica}

---

### 2.2 Data Preprocessing

#### Label Encoding
Categorical labels were encoded numerically as:
- Setosa → 1
- Versicolor → 2
- Virginica → 3

#### Train-Test Split
The dataset was split randomly:
- **Training Set:** 80%
- **Testing Set:** 20%

This ensures unbiased performance evaluation.

---

## 3. Mathematical Formulation of the Model

Two different models were implemented to analyze performance differences.

---

### 3.1 Multinomial Logistic Regression (Softmax Regression)

Let:
- \( x \in \mathbb{R}^4 \) be the input feature vector
- \( W \in \mathbb{R}^{4 \times 3} \) be the weight matrix
- \( b \in \mathbb{R}^3 \) be the bias vector

#### Logits
\[
z_k = w_k^T x + b_k
\]

#### Softmax Function
\[
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{3} e^{z_j}}
\]

---

### 3.2 Linear Regression (Baseline Model)

The regression model predicts a scalar output:
\[
h(x) = w^T x + b
\]

Predicted values are rounded to obtain discrete class labels.  
This approach imposes an artificial ordinal structure on nominal classes.

---

## 4. Loss Function and Training Process

### 4.1 Logistic Regression Loss

The **categorical cross-entropy loss** is minimized:
\[
L = -\sum_{k=1}^{3} y_k \log(\hat{y}_k)
\]

With L2 regularization:
\[
J(W) = \frac{1}{m} \sum_{i=1}^{m} L^{(i)} + \frac{\lambda}{2} \|W\|_F^2
\]

#### Optimization
- Optimizer: **L-BFGS**
- Type: Quasi-Newton Method
- Advantage: Fast convergence for small datasets

---

### 4.2 Linear Regression Loss

The **Residual Sum of Squares (RSS)** is minimized:
\[
J(w) = \sum_{i=1}^{m} (y^{(i)} - h(x^{(i)}))^2
\]

Closed-form solution:
\[
w = (X^T X)^{-1} X^T y
\]

---

## 5. Model Architecture and Justification

### 5.1 Multinomial Logistic Regression

**Architecture**
- Input Layer: 4 features
- Linear Transformation
- Softmax Output Layer (3 neurons)

**Justification**
- Designed for multi-class classification
- Produces probabilistic outputs
- Interpretable and computationally efficient

---

### 5.2 Linear Regression

**Architecture**
- Input Layer
- Single linear output neuron

**Justification**
- Included as a baseline model
- Highlights limitations of regression for classification
- Demonstrates model–task mismatch

---

## 6. Evaluation Methodology and Results

### Evaluation Metrics
- Accuracy
- Confusion Matrix

### Methodology
- Identical train-test splits for both models
- Evaluation on unseen test data

### Results
- Logistic Regression achieved high accuracy
- Perfect classification of Setosa
- Minor confusion between Versicolor and Virginica
- Linear Regression showed inferior performance

---

## 7. Limitations and Future Improvements

### Limitations
1. Limited dataset size
2. Synthetic data may introduce bias
3. Linear decision boundaries restrict expressiveness
4. Linear Regression unsuitable for nominal classes

---

### Future Improvements
1. Use non-linear classifiers (SVM, Decision Trees)
2. Apply k-fold cross-validation
3. Feature scaling and PCA
4. Evaluate precision, recall, and F1-score
5. Explore neural network classifiers

---

## Conclusion

This project demonstrates the importance of selecting models aligned with the underlying problem structure. Multinomial Logistic Regression provides a principled solution for multi-class classification, while Linear Regression serves as a baseline highlighting conceptual limitations. The study reinforces foundational machine learning concepts and sets the stage for more advanced exploration.
