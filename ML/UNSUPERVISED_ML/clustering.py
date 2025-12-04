import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering

iris = load_iris()
X = iris.data[:, :2]    # 2D for simple understanding/plotting (if you want)
X_samples = X.shape[0]
X_clusters = 3

print(f"\nDataset: Iris (first 2 features), Samples: {X_samples}, Features: {X.shape[1]}")
print(f"Number of clusters (k): {X_clusters}")
print("Feature Names:", iris.feature_names)
print("Label Names:", iris.target_names)

# Generate sample clustering dataset
X_cluster, y_cluster = make_blobs(n_samples=300, centers=3, n_features=2, 
                                cluster_std=0.6, random_state=42)

# 3.1 k-Means Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_cluster)
print(f"k-Means completed with {len(np.unique(kmeans_labels))} clusters")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

# 3.2 Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_cluster)
print(f"Hierarchical Clustering completed with {len(np.unique(hierarchical_labels))} clusters")

# 3.3 DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")