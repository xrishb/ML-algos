# Logistic Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()

# Create a DataFrame with the features
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
print("\nFeature preview:")
print(X.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["malignant", "benign"])
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Feature importance visualization
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Visualize top 10 feature importance
plt.subplot(2, 1, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('logistic_regression_results.png')
print("\nResults saved as 'logistic_regression_results.png'")

if __name__ == "__main__":
    print("\nLogistic regression analysis complete!") 