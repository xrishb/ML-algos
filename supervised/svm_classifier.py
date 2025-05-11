# Support Vector Machine (SVM) Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()

# Create a DataFrame with the features
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Map target names for better readability
target_names = data.target_names
y_names = pd.Series([target_names[i] for i in y], name='diagnosis')

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {pd.Series(y_names).value_counts().to_dict()}")

print("\nData preview:")
print(X.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
print("\nTraining SVM model...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png')
print("\nConfusion matrix saved as 'svm_confusion_matrix.png'")

# Hyperparameter tuning using GridSearchCV
print("\nPerforming hyperparameter tuning with GridSearchCV...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Use a subset of the data for faster grid search
grid_X = X_train_scaled[:500]  # Use a subset for faster demo
grid_y = y_train[:500]

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Run grid search
print("\nThis may take some time...")
grid_search.fit(grid_X, grid_y)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the model with the best parameters on the full training set
print("\nTraining SVM with best parameters...")
best_svm = SVC(**grid_search.best_params_, random_state=42)
best_svm.fit(X_train_scaled, y_train)

# Make predictions with the optimized model
best_y_pred = best_svm.predict(X_test_scaled)

# Evaluate the optimized model
best_accuracy = accuracy_score(y_test, best_y_pred)
best_report = classification_report(y_test, best_y_pred, target_names=target_names)
best_conf_matrix = confusion_matrix(y_test, best_y_pred)

print("\nOptimized model evaluation:")
print(f"Accuracy: {best_accuracy:.4f}")
print("\nClassification Report:")
print(best_report)

# Compare model performance before and after tuning
print("\nPerformance Comparison:")
print(f"Original SVM accuracy: {accuracy:.4f}")
print(f"Optimized SVM accuracy: {best_accuracy:.4f}")
print(f"Improvement: {(best_accuracy - accuracy) * 100:.2f}%")

# Visualize optimized model confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized SVM Confusion Matrix')
plt.tight_layout()
plt.savefig('svm_optimized_confusion_matrix.png')
print("\nOptimized model confusion matrix saved as 'svm_optimized_confusion_matrix.png'")

if __name__ == "__main__":
    print("\nSVM analysis complete!") 