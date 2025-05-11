# Random Forest 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_openml

# Load the mnist_784 dataset (handwritten digits)
print("Loading MNIST dataset (this may take a moment)...")
try:
    # Try to load a small subset for demonstration
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto', 
                        cache=True, return_X_y=True)
    X, y = mnist
    
    # Take a subset for faster processing
    n_samples = 10000
    X = X.iloc[:n_samples]
    y = y.iloc[:n_samples]
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating a synthetic dataset instead...")
    # Create a synthetic dataset if MNIST fails to load
    from sklearn.datasets import make_classification
    X, y_numeric = make_classification(n_samples=10000, n_features=20, 
                                     n_informative=10, n_classes=10, 
                                     random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series([str(label) for label in y_numeric])

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(y.unique())}")
print(f"Class distribution: {y.value_counts().head(3)} ... (showing top 3)")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the random forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=15,      # Maximum depth of trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Train the model
model.fit(X_train, y_train)

# Cross-validation to assess model stability
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Plot feature importance
if X.shape[1] <= 30:  # Only show if we have a reasonable number of features
    n_features = min(20, X.shape[1])  # Show at most top 20 features
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    plt.figure(figsize=(12, 8))
    plt.title('Random Forest Feature Importances')
    plt.bar(range(n_features), importances[indices], align='center')
    plt.xticks(range(n_features), X.columns[indices], rotation=90)
    plt.xlim([-1, n_features])
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')
    print("\nFeature importances plot saved as 'random_forest_feature_importance.png'")
    
    # Print top features
    print("\nTop features by importance:")
    for i, idx in enumerate(indices[:10]):  # Show top 10
        print(f"{i+1}. {X.columns[idx]}: {importances[idx]:.4f}")

# Plot confusion matrix for a subset of classes (if we have many)
plt.figure(figsize=(10, 8))
unique_classes = np.unique(y_test)
n_classes_to_plot = min(10, len(unique_classes))  # Plot at most 10 classes
cm = confusion_matrix(y_test, y_pred, labels=unique_classes[:n_classes_to_plot])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_classes[:n_classes_to_plot],
            yticklabels=unique_classes[:n_classes_to_plot])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('random_forest_confusion_matrix.png')
print("Confusion matrix saved as 'random_forest_confusion_matrix.png'")

if __name__ == "__main__":
    print("\nRandom Forest analysis complete!") 