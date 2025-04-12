# Logistic Regression & SVM Classifier Comparison

## Description
This project compares the performance of Logistic Regression and Support Vector Machine (SVM) classifiers on a non-linear dataset. Using scikit-learn's `make_moons` dataset, it trains both models with default parameters and visualizes their decision boundaries while displaying training/test accuracies.

## Features
- Generates synthetic moons dataset with 10,000 samples and 0.4 noise
- Splits data into training (80%) and test (20%) sets
- Implements two classifier types:
  - Logistic Regression (linear model)
  - SVM with RBF kernel (non-linear model)
- Visualizes decision boundaries and data points in side-by-side comparisons
- Highlights fundamental differences in linear vs non-linear model capabilities

## Installation
```bash
pip install numpy matplotlib scikit-learn
