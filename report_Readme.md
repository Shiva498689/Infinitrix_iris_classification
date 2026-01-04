# Infinitrix AI/ML Domain Induction Task Report

## Iris Species Classification Using Morphological Features

**Author:** Shiva Dubey  
**Roll Number:** 250002069  
**Branch:** Electrical Engineering  
**Email:** ee250002069@iiti.ac.in  

---

## Abstract

This project presents a supervised machine learning approach for classifying Iris flower species using morphological measurements. Two models are evaluated: **Multinomial Logistic Regression** and **Linear Regression** adapted for classification. The objective is to analyze model suitability, mathematical formulation, training dynamics, and performance limitations. Experimental results for this particular classification problem both the models achieved the same level of accuracy , but generally , Logistic regression models outperform the Linear Regression in Classification Problems  show that probabilistic classifiers outperform regression-based approaches for nominal multi-class problems. This study reinforces foundational principles of model selection and loss-function alignment in machine learning.

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
- **Features:** 4 continuous variables  
- **Classes:** 3 (Setosa, Versicolor, Virginica)  

**Feature Vector:**

$$
x = [\text{sepal length, sepal width, petal length, petal width}]
$$


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

- Input Layer: 4 features   
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
- **Linear Regression:** High accuracy;  perfect Setosa classification; minor confusion between Versicolor and Virginica
- ** Overall Accuracy = 97.56 % of both the models
  *Generally this do not happens because Linear Classification is made for predicting the linear outputs rather than Classification purposes  

---

## 7. Limitations and Future Improvements

### 7.1 Limitations

1. Small dataset size limits generalization
2. Linearly separable data hence easy to make detection perfection
3. model memorizes only single rule to differentiate between the species
4. Linear Regression model not suitable for such models because they do not outputs probabilities in comparison to
   Logistic Regression which uses softmax for giving probabilities as outputs

  

### 7.2 Future Improvements

1. Evaluate precision, recall, and F1-score  
2. Explore neural network-based models
3. Train the models on Actual images instead of length data
4. To use CNNs on images of Iris flower to build industry grade solution
5. Use Transfer Learning for improving accuracy on images of flowers  

---

## 8. Conclusion

Multinomial Logistic Regression provides a principled solution for multi-class classification, while Linear Regression serves as a baseline highlighting conceptual limitations. This project reinforces key principles of model selection, loss functions, and evaluation methodology in supervised learning.

## 9. Confusion Matrix of the models


 




