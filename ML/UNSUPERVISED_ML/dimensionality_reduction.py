import numpy as np
# Generate high-dimensional dataset
X_dimred = np.random.randn(200, 10)  # 200 samples, 10 features   #AMAN KUMAR

# 4.1 Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dimred)
print(f"Original shape: {X_dimred.shape}")
print(f"PCA reduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# 4.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_dimred)
print(f"Original shape: {X_dimred.shape}")
print(f"t-SNE reduced shape: {X_tsne.shape}")