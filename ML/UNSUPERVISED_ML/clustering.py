# Clustering Techniques: k-Means, Hierarchical Clustering, DBSCAN
# Dataset: Synthetic dataset generated using make_blobs()

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# -------------------------------------------------------
# 1. Generate synthetic dataset for clustering
# -------------------------------------------------------
X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=0.6,
    random_state=42
)

print("\nDataset Information:")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("True Clusters:", len(np.unique(y_true)))

# -------------------------------------------------------
# 2. K-Means Clustering
# -------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

print("\nK-Means Clustering:")
print("Clusters found:", len(np.unique(kmeans_labels)))
print("Cluster Centers:\n", kmeans.cluster_centers_)

# -------------------------------------------------------
# 3. Hierarchical (Agglomerative) Clustering
# -------------------------------------------------------
hier = AgglomerativeClustering(n_clusters=3)
hier_labels = hier.fit_predict(X)

print("\nHierarchical Clustering:")
print("Clusters found:", len(np.unique(hier_labels)))

# -------------------------------------------------------
# 4. DBSCAN Clustering
# -------------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_points = list(dbscan_labels).count(-1)

print("\nDBSCAN Clustering:")
print("Clusters found:", n_clusters_dbscan)
print("Noise points:", n_noise_points)
