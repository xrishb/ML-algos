# K-means Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine

# Load the wine dataset
print("Loading Wine dataset...")
wine = load_wine()

# Create a DataFrame with the features
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of actual classes: {len(np.unique(y))}")

print("\nData preview:")
print(X.head())

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the optimal number of clusters using the elbow method
print("\nFinding optimal number of clusters using the Elbow Method...")
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    print(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_score(X_scaled, labels):.4f}")

# Plotting the Elbow Method results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method for Optimal k')
plt.grid(True)

plt.tight_layout()
plt.savefig('kmeans_elbow_method.png')
print("\nElbow method plot saved as 'kmeans_elbow_method.png'")

# Choose the number of clusters based on elbow method and silhouette scores
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Apply K-means clustering with the optimal k
print(f"\nApplying K-means clustering with k={optimal_k}...")
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_optimal.fit_predict(X_scaled)

# Add the cluster labels to the original data
X_with_clusters = X.copy()
X_with_clusters['cluster'] = clusters
X_with_clusters['actual_class'] = y

# Show the distribution of actual classes within each cluster
print("\nCluster composition (actual classes in each cluster):")
for cluster in range(optimal_k):
    cluster_data = X_with_clusters[X_with_clusters['cluster'] == cluster]
    class_counts = cluster_data['actual_class'].value_counts()
    print(f"\nCluster {cluster} size: {len(cluster_data)}")
    for cls, count in class_counts.items():
        print(f"  Class {cls} ({wine.target_names[cls]}): {count} samples ({count/len(cluster_data)*100:.1f}%)")

# Visualize the clusters using PCA for dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))

# Plot the clusters
plt.subplot(1, 2, 1)
for cluster in range(optimal_k):
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], 
                label=f'Cluster {cluster}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering Results (PCA View)')
plt.legend()
plt.grid(True)

# Plot the actual classes
plt.subplot(1, 2, 2)
for i, class_name in enumerate(wine.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                label=class_name, alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Actual Classes (PCA View)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('kmeans_clustering_results.png')
print("\nClustering results visualization saved as 'kmeans_clustering_results.png'")

# Print the cluster centers
print("\nCluster Centers (top 3 features for each cluster):")
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans_optimal.cluster_centers_),
    columns=wine.feature_names
)

for cluster in range(optimal_k):
    print(f"\nCluster {cluster} center (selected features):")
    # Sort features by absolute distance from the overall mean
    feature_importance = pd.Series({
        feature: abs(cluster_centers.loc[cluster, feature] - X[feature].mean())
        for feature in wine.feature_names
    }).sort_values(ascending=False)
    
    for feature in feature_importance.index[:3]:
        print(f"  {feature}: {cluster_centers.loc[cluster, feature]:.2f} " +
              f"(dataset avg: {X[feature].mean():.2f})")

if __name__ == "__main__":
    print("\nK-means clustering analysis complete!") 