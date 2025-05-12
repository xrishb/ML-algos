# Machine Learning Algorithms

This repository contains implementations of various machine learning algorithms using scikit-learn, pandas, and numpy.

## Supervised Learning

Each file demonstrates a different algorithm and can be run independently:

- `linear_regression.py`: Linear Regression implementation
- `logistic_regression.py`: Logistic Regression for classification
- `decision_tree.py`: Decision Tree classifier
- `random_forest.py`: Random Forest ensemble method
- `svm_classifier.py`: Support Vector Machine
- `naive_bayes.py`: Naive Bayes classifier

## Unsupervised Learning

- `kmeans_clustering.py`: K-means clustering
- `pca_dimension_reduction.py`: Principal Component Analysis

## Reinforcement Learning

- `reinforcement_learning/q_learning.py`: Q-Learning algorithm implementation with a GridWorld environment

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

To install requirements:
```
pip install -r requirements.txt
```

## Running the Examples

Each algorithm can be run independently:

```
python linear_regression.py
python reinforcement_learning/q_learning.py