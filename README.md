# Machine-Learning-Algorithms-Comparison

This project focuses on implementing and evaluating core machine learning algorithms, including K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), and ensemble methods such as Boosting and Bagging. The goal is to deepen understanding of these techniques, experiment with configurations, and evaluate performance using standard classification metrics.

## Objective

The assignment aims to:
- Implement and evaluate machine learning algorithms.
- Experiment with different configurations and kernels.
- Compare individual models with ensemble methods.
- Use classification metrics to assess model performance.

## Dataset

Use a publicly available classification dataset, such as:
- Breast Cancer Dataset

## Tools and Libraries

The following Python APIs and libraries will be used:
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Evaluation Metrics

The following metrics will be used to evaluate model performance:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Assignment Tasks

### Part 1: K-Nearest Neighbors (KNN)
- Implement the KNN algorithm using APIs.
- Experiment with at least three distance metrics:
  - Euclidean Distance
  - Manhattan Distance
  - Cosine Distance
- Use cross-validation to determine the optimal value of **K** (number of neighbors).
- Analyze and discuss:
  - The effect of distance metrics on classification performance.
  - The best value of **K** and its justification.

### Part 2: Logistic Regression
- Train a Logistic Regression model on the dataset.
- Experiment with different regularization techniques:
  - L1 Regularization
  - L2 Regularization
- Evaluate the model using classification metrics.
- Compare Logistic Regression's performance with KNN.

### Part 3: Support Vector Machines (SVM)
- Implement SVM using APIs.
- Train the model with at least three kernels:
  - Linear Kernel
  - Polynomial Kernel
  - Radial Basis Function (RBF) Kernel
- Evaluate and compare kernel performance using classification metrics.
- Discuss the impact of kernel choice on accuracy and other metrics.

### Part 4: Ensemble Methods
- **Boosting:** Train a model using AdaBoost.
- **Bagging:** Train a model using Bagging or Random Forest.
- Compare the performance of Boosting and Bagging methods.
- Discuss:
  - Which ensemble method performed better and why.
  - How ensemble methods compare to individual models (KNN, Logistic Regression, SVM).

## Deliverables

### Code
- Submit a well-documented Jupyter Notebook or Python script with the implementation.

### Report
- A concise report (9 pages) that includes:
  - Introduction to each method.
  - Experimental approach and algorithm implementation.
  - Detailed analysis of results and metric comparisons.
  - Conclusions and findings.
