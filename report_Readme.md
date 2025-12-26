# Infinitrix AI/ML Domain Induction Task Report

**Topic:** Iris Classifier using Morphological Features  
**Author:** Shiva Dubey  
**Roll Number:** 250002069  
**Branch:** Electrical Engineering  
**Email:** ee250002069@iiti.ac.in  

---

## Project Overview

**Framework:** Scikit-Learn  
**Environment:** Google Colab  
**Input Vector:** $\{ \text{sepal length, sepal width, petal length, petal width} \}$  
**Output Classes:** $\{ \text{Setosa, Versicolor, Virginica} \}$

### Introduction
This project explores the implementation of supervised learning algorithms to classify Iris species based on morphological features. Two distinct models are evaluated:
1.  **Multinomial Logistic Regression** (Softmax Regression)
2.  **Linear Regression** (Adapted for Classification via Label Encoding)

Both models are trained on labeled data provided in `iris.csv` to learn the mapping function $f: X \to Y$.

---

## Dataset Description

The primary dataset is derived from the standard Iris dataset (originally hosted on Kaggle).
* **Original Size:** 150 data points.
* **Augmentation:** The dataset has been augmented with synthetic data points generated via AI methods, bringing the total sample size to approximately 160.
* **Feature Space:** The dataset consists of 5 columns. The first four are continuous floating-point features:
    * Sepal Length
    * Sepal Width
    * Petal Length
    * Petal Width
* **Target Variable:** The 5th column, `specie`, serves as the categorical label.

---

## Workflow Methodology

The data processing pipeline follows these steps:

1.  **Label Encoding:**
    The categorical string labels are mapped to numerical ordinal values to facilitate mathematical processing:
    * Setosa $\rightarrow 1$
    * Versicolor $\rightarrow 2$
    * Virginica $\rightarrow 3$

2.  **Train-Test Split:**
    To ensure robust model evaluation and prevent overfitting, the dataset is partitioned using a randomized split strategy.
    * **Training Set:** 80% of the data (used for parameter optimization).
    * **Testing Set:** 20% of the data (used for validation).

---

## Mathematical Foundations of Iris Classification

This section provides a rigorous analysis of the hypothesis spaces, objective functions, and optimization strategies employed "under the hood" by Scikit-Learn.

### 1. Model I: Multinomial Logistic Regression

Logistic Regression is used here as a probabilistic classifier. Since there are $K=3$ classes, the model generalizes to **Softmax Regression**.

#### 1.1 The Forward Pass (Hypothesis)
The model parameterizes a linear transformation followed by a non-linear activation. Let $W \in \mathbb{R}^{n \times K}$ be the weight matrix and $b \in \mathbb{R}^K$ be the bias vector.
For an input vector $x$, we compute the **logits** $z_k = w_k^T x + b_k$. To interpret these as probabilities, we apply the **Softmax Function** $\sigma: \mathbb{R}^K \to \Delta^{K-1}$:

$$P(y=k \mid x; W, b) = \hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

This ensures that $\hat{y}_k \in [0, 1]$ and $\sum_{k=1}^K \hat{y}_k = 1$.

#### 1.2 The Objective Function (Loss)
We minimize the **Categorical Cross-Entropy Loss** (Log-Loss). For a single example $(x, y)$, let $y$ be one-hot encoded. The loss $L$ is:

$$L(W, b) = - \sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

The global cost function $J$ over $m$ samples, including $L_2$ regularization (Ridge), is:

$$J(W, b) = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \mathbb{I}(y^{(i)}=k) \log(\hat{y}_k^{(i)}) + \frac{\lambda}{2} \|W\|_F^2$$

#### 1.3 Optimization (Backward Pass)
The gradients with respect to the weights are computed via the Chain Rule:

$$\frac{\partial J}{\partial w_k} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_k^{(i)} - y_k^{(i)}) x^{(i)} + \lambda w_k$$

Scikit-Learn optimizes this using the **L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) algorithm, a Quasi-Newton method that approximates the Hessian matrix for faster convergence.

---

### 2. Model II: Linear Regression (OLS)



Linear Regression treats the classification problem as a continuous regression task, predicting a scalar value rather than a probability distribution.

#### 2.1 The Forward Pass
The hypothesis is an affine transformation mapping $\mathbb{R}^n \to \mathbb{R}$:
$$h(x) = w^T x + b$$

#### 2.2 The Objective Function
The model minimizes the **Residual Sum of Squares (RSS)**:
$$J(w, b) = \sum_{i=1}^{m} (y^{(i)} - h(x^{(i)}))^2 = \| y - Xw \|_2^2$$

*Note: Applying OLS to classification imposes an ordinal relationship (e.g., treating Class 3 as "greater than" Class 1), which introduces bias for nominal classes.*

#### 2.3 Closed-Form Solution
The optimal weights are found via the **Normal Equation**:
$$w = (X^T X)^{-1} X^T y$$

---

### 3. Evaluation Metrics

#### 3.1 Confusion Matrix
The confusion matrix $C \in \mathbb{N}^{K \times K}$ allows visualization of model performance:
$$C_{ij} = \sum_{i=1}^m \mathbb{I}(y^{(i)}_{true} = i \land y^{(i)}_{pred} = j)$

#### 3.2 Accuracy Score
The accuracy metric calculates the proportion of correct predictions:
$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\hat{y}^{(i)} = y^{(i)})$$