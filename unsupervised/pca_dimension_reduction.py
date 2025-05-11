# Principal Component Analysis (PCA) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the wine dataset
print("Loading Wine dataset...")
wine = load_wine()

# Create a DataFrame with the features
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")

print("\nFirst few rows of the dataset:")
print(X.head())

# Standardize the features (important for PCA)
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
print("\nApplying PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance ratio per component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Plot explained variance
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8)
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'r-', marker='o')
plt.axhline(y=0.9, color='g', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'b-', marker='o')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.axhline(y=0.95, color='g', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_explained_variance.png')
print("\nExplained variance plot saved as 'pca_explained_variance.png'")

# Find the number of components needed to explain 95% of variance
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"\nNumber of components needed to explain 95% of variance: {n_components_95}")

# Apply PCA with the reduced number of components
pca_95 = PCA(n_components=n_components_95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"\nData shape after PCA reduction: {X_pca_95.shape}")
print(f"Dimension reduction: {X.shape[1]} -> {X_pca_95.shape[1]} features")

# Visualize first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='k')
plt.colorbar(scatter, label='Wine Type')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Wine Dataset - First Two Principal Components')
plt.grid(True)
plt.savefig('pca_visualization.png')
print("\nPCA visualization saved as 'pca_visualization.png'")

# Feature Contribution analysis
feature_contributions = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=wine.feature_names
)

# Plot heatmap of feature contributions to principal components
plt.figure(figsize=(12, 10))
sns.heatmap(feature_contributions.iloc[:, :5], annot=True, cmap='coolwarm', center=0)
plt.title('Feature Contributions to First 5 Principal Components')
plt.tight_layout()
plt.savefig('pca_feature_contributions.png')
print("\nFeature contributions heatmap saved as 'pca_feature_contributions.png'")

# Compare model performance with and without PCA
print("\nComparing model performance with and without PCA...")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

X_train_pca = pca_95.transform(X_train)
X_test_pca = pca_95.transform(X_test)

# Create and train a classifier on original data
print("\nTraining Random Forest on original data...")
rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_original.fit(X_train, y_train)
y_pred_original = rf_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# Create and train a classifier on PCA-reduced data
print("\nTraining Random Forest on PCA-reduced data...")
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Compare results
print("\nModel Comparison Results:")
print(f"Accuracy with original {X.shape[1]} features: {accuracy_original:.4f}")
print(f"Accuracy with {n_components_95} principal components: {accuracy_pca:.4f}")
print(f"Difference: {(accuracy_pca - accuracy_original):.4f}")

print("\nClassification Report - Original Features:")
print(classification_report(y_test, y_pred_original, target_names=wine.target_names))

print("\nClassification Report - PCA Features:")
print(classification_report(y_test, y_pred_pca, target_names=wine.target_names))

# Visualization in 3D (using the first 3 principal components)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=y, cmap='viridis', s=50, alpha=0.8, edgecolors='k'
)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Wine Dataset - First Three Principal Components')

plt.colorbar(scatter, label='Wine Type')
plt.tight_layout()
plt.savefig('pca_3d_visualization.png')
print("\n3D PCA visualization saved as 'pca_3d_visualization.png'")

if __name__ == "__main__":
    print("\nPCA dimension reduction analysis complete!") 