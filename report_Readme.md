# Infinitrix AI/ML Domain Induction Task Report

## Iris Species Classification Using Morphological Features

**Author:** Shiva Dubey  
**Roll Number:** 250002069  
**Branch:** Electrical Engineering  
**Email:** ee250002069@iiti.ac.in  

---

## Abstract

This project presents a supervised machine learning approach for classifying Iris flower species using morphological measurements. Two models are evaluated: **Multinomial Logistic Regression** and **Linear Regression** adapted for classification. The objective is to analyze model suitability, mathematical formulation, training dynamics, and performance limitations. Experimental results show that probabilistic classifiers outperform regression-based approaches for nominal multi-class problems. This study reinforces foundational principles of model selection and loss-function alignment in machine learning.

---

**Keywords:** Iris Dataset, Multi-class Classification, Logistic Regression, Softmax, Machine Learning

---

## 1. Problem Overview and Motivation

Classification of biological species based on physical characteristics is a fundamental problem in pattern recognition and supervised learning. The Iris dataset serves as a benchmark due to its balanced structure and partially linearly separable feature space.

**Motivation:**
- Understand supervised classification from first principles
- Compare appropriate classifiers against naïve baselines
- Analyze the mathematical foundations of loss functions and optimization
- Build an end-to-end ML workflow using Scikit-Learn

---

## 2. Dataset Description and Preprocessing

### 2.1 Dataset Description

- **Original Samples:** 150  
- **Augmented Samples:** ~160  
- **Features:** 4 continuous variables  
- **Classes:** 3 (Setosa, Versicolor, Virginica)  

**Feature Vector:**

\[
x = [ \text {sepal length, sepal width, petal length, petal width} \text]
\]

**Target Variable:** `specie` ∈ {Setosa, Versicolor, Virginica}

---

### 2.2 Preprocessing

**Label Encoding:**
- Setosa → 1  
- Versicolor → 2  
- Virginica → 3  

**Train-Test Split:**
- Training Set: 80%  
- Testing Set: 20%  

---

## 3. Mathematical Formulation of the Model

### 3.1 Multinomial Logistic Regression

Let \( x \in \mathbb{R}^4 \) be the input vector, \( W \in \mathbb{R}^{4 \times 3} \) the weight matrix, and \( b \in \mathbb{R}^3 \) the bias vector.

**Logits:**

$$
z_k = w_k^T x + b_k
$$

**Softmax Function:**

$$
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{3} e^{z_j}}
$$

---

### 3.2 Linear Regression (Baseline)

The regression model predicts a scalar value:

$$
h(x) = w^T x + b
$$

Predicted outputs are rounded to the nearest integer to get class labels.  
This imposes an artificial ordinal structure on nominal classes.

---

## 4. Loss Function and Training Process

### 4.1 Logistic Regression Loss

The **categorical cross-entropy loss** is:

$$
L = - \sum_{k=1}^{3} y_k \log(\hat{y}_k)
$$

Including **L2 regularization**:

$$
J(W) = \frac{1}{m} \sum_{i=1}^{m} L^{(i)} + \frac{\lambda}{2} \| W \|_F^2
$$

**Optimization:** L-BFGS (quasi-Newton method).

---

### 4.2 Linear Regression Loss

The **Residual Sum of Squares (RSS)**:

$$
J(w) = \sum_{i=1}^{m} (y^{(i)} - h(x^{(i)}))^2
$$

**Closed-form solution (Normal Equation):**

$$
w = (X^T X)^{-1} X^T y
$$

---

## 5. Model Architecture and Justification

### 5.1 Logistic Regression

- Input Layer: 4 features  
- Linear Transformation  
- Softmax Output Layer (3 neurons)  

**Justification:**  
- Suitable for multi-class classification  
- Produces probabilistic outputs  
- Interpretable and computationally efficient  

---

### 5.2 Linear Regression

- Input Layer  
- Single linear output neuron  

**Justification:**  
- Baseline to show limitations of regression for classification  
- Demonstrates model-task mismatch  

---

## 6. Evaluation Methodology and Results

### 6.1 Evaluation Metrics

- Accuracy  
- Confusion Matrix  

### 6.2 Methodology

- Randomized 80:20 train-test split  
- Both models trained on the same training data  
- Evaluated on unseen test data  

### 6.3 Results

- **Logistic Regression:** High accuracy; perfect Setosa classification; minor confusion between Versicolor and Virginica  
- **Linear Regression:** Lower accuracy; misclassifications due to inappropriate ordinal assumptions  

---

## 7. Limitations and Future Improvements

### 7.1 Limitations

1. Small dataset size limits generalization  
2. Synthetic data may introduce bias  
3. Linear decision boundaries restrict expressiveness  
4. Linear Regression unsuitable for nominal classes  

### 7.2 Future Improvements

1. Use non-linear classifiers (SVM, Decision Trees)  
2. Apply k-fold cross-validation  
3. Feature scaling and PCA  
4. Evaluate precision, recall, and F1-score  
5. Explore neural network-based models  

---

## 8. Conclusion

Multinomial Logistic Regression provides a principled solution for multi-class classification, while Linear Regression serves as a baseline highlighting conceptual limitations. This project reinforces key principles of model selection, loss functions, and evaluation methodology in supervised learning.

---

## References

1. R. A. Fisher, “The Use of Multiple Measurements in Taxonomic Problems,” *Annals of Eugenics*, 1936.  
2. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, Springer.
