# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer

# # Load Breast Cancer dataset (binary classification)
# data_class = load_breast_cancer()
# x_class=data_class.data
# y_class=data_class.target

# print("Features shape:", x_class.shape)
# print("Labels shape:", y_class.shape)

# # Train-test split
# x_train, x_test, y_train, y_test=train_test_split(
#     x_class, y_class, test_size=0.3, random_state=42, stratify=y_class 
#     )

# # 1.1 Logistic Regression
# from sklearn.linear_model import LogisticRegression
# log_reg=LogisticRegression(random_state=42, max_iter=1000 )
# log_reg.fit(x_train,y_train)
# log_accuracy=log_reg.score(x_test, y_test)
# print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")


# #1.2 Support Vector Machine (SVM)
# from sklearn.svm import SVC
# svm=SVC(kernel='rbf',random_state=42)
# svm.fit(x_train, y_train)
# svm_accuracy=svm.score(x_test, y_test)
# print(f"SVM Accuracy: {svm_accuracy:.4f} ")

# #1.3 Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# D_tree=DecisionTreeClassifier(max_depth=5, random_state=42)
# D_tree.fit(x_train, y_train)
# D_tree_accuracy=D_tree.score(x_test, y_test)
# print(f"Decision Tree Accuracy: {D_tree_accuracy:.4f} ")


# #1.4 Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# nb=GaussianNB()
# nb.fit(x_train, y_train)
# nb_accuracy=nb.score(x_test, y_test)
# print(f"naive bayes accuracy: {nb_accuracy:.4f}")

# #1.5 k-Nearest Neighbors (KNN)
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train, y_train)
# knn_accuracy=knn.score(x_test, y_test)
# print(f"knn accuracy: {knn_accuracy:.4f}")

# print("----------------------------------------")

# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split

# # Load Diabetes dataset (regression)
# data_class=load_diabetes()
# x_class=data_class.data
# y_class=data_class.target

# # train-twest split
# x_train, x_test, y_train, y_test=train_test_split(
#     x_class, y_class, test_size=0.2,random_state=42
#     )

# # 2.1 Linear Regression
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge,Lasso
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

# lin_reg=LinearRegression()
# lin_reg.fit(x_train, y_train)
# y_pred_lin=lin_reg.predict(x_test)
# lin_r2=r2_score(y_test, y_pred_lin)
# lin_mse=mean_squared_error(y_test, y_pred_lin)
# lin_mas=mean_absolute_error(y_test, y_pred_lin)
# lin_rmse=root_mean_squared_error(y_test, y_pred_lin)
# print(f"linear regression r2 score: {lin_r2:.4f}")
# print(f"linear regression mse: {lin_mse:.4f}")
# print(f"linear regression mae: {lin_mas:.4f}")
# print(f"linear regression rmse: {lin_rmse:.4f}")


# # 2.2 Ridge Regression
# ridge_reg=Ridge(alpha=1.0)
# ridge_reg.fit(x_train, y_train)
# y_pred_ridge=ridge_reg.predict(x_test)
# ridge_r2=r2_score(y_test,y_pred_ridge)
# ridge_mse=mean_squared_error(y_test,y_pred_ridge)
# ridge_mae=mean_absolute_error(y_test,y_pred_ridge)
# ridge_rmse=root_mean_squared_error(y_test,y_pred_ridge)
# print(f"ridge regression r2 score: {ridge_r2:.4f}")
# print(f"ridge regression mse: {ridge_mse:.4f}")
# print(f"ridge regression mae: {ridge_mae:.4f}")
# print(f"ridge regression rmse: {ridge_rmse:.4f}")

# # 2.3 Lasso Regression
# lasso_reg=Lasso(alpha=1.0)
# lasso_reg.fit(x_train, y_train)
# y_pred_lasso=lasso_reg.predict(x_test)
# lasso_r2=r2_score(y_test,y_pred_lasso)
# lasso_mse=mean_squared_error(y_test,y_pred_lasso)
# lasso_mae=mean_absolute_error(y_test,y_pred_lasso)
# lasso_rmse=root_mean_squared_error(y_test,y_pred_lasso)
# print(f"lasso regression r2 score: {lasso_r2:.4f}")
# print(f"lasso regression mse: {lasso_mse:.4f}")
# print(f"lasso regression mae: {lasso_mae:.4f}")
# print(f"lasso regression rmse: {lasso_rmse:.4f}")

# # 2.4 Support Vector Regression (SVR)
# from sklearn.svm import SVR
# svr=SVR(kernel='rbf')
# svr.fit(x_train, y_train)
# svr_pred=svr.predict(x_test)
# svr_r2_score=r2_score(y_test,svr_pred)
# svr_mse=mean_squared_error(y_test,svr_pred)
# svr_mae=mean_absolute_error(y_test,svr_pred)
# svr_rmse=root_mean_squared_error(y_test,svr_pred)
# print(f"svr r2 score: {svr_r2_score:.4f}")
# print(f"svr mse: {svr_mse:.4f}")
# print(f"svr mae: {svr_mae:.4f}")
# print(f"svr rmse: {svr_rmse:.4f}")

# print("---------------------------------------------------")

# Clustering Techniques: k-Means, Hierarchical Clustering, DBSCAN
# Dataset: Synthetic dataset generated using make_blobs()
# import numpy as np
# from sklearn.datasets import make_blobs

# x, y_true=make_blobs(
#     n_samples=300,
#     n_features=2,
#     centers=3,
#     cluster_std=0.6,
#     random_state=42
# )

# print("\ndataset information: ")
# print("samples: ",x.shape[0])
# print("features: ",x.shape[1])
# print("true clusters: ",len(np.unique(y_true)))

# # 2. K-Means Clustering
# from sklearn.cluster import KMeans

# kmeans=KMeans(n_clusters=3, random_state=42)
# kmeans_labels=kmeans.fit_predict(x)

# print("\nk-means clustering: ")
# print("clusters found: ", len(np.unique(kmeans_labels)))
# print("cluster centers:\n ",kmeans.cluster_centers_)

# # 3. Hierarchical (Agglomerative) Clustering
# from sklearn.cluster import AgglomerativeClustering

# hier=AgglomerativeClustering(n_clusters=3)
# hier_labels=hier.fit_predict(x)

# print("\nhierarchical clustering: ")
# print("clusters found :", len(np.unique(hier_labels)))

# print("1-------------------------------------------------------")
# import numpy as np

# x_dimred=np.random.randn(100,10)
# print(x_dimred)

# print("2-------------------------------------------------------")
# from sklearn.decomposition import PCA
# pca=PCA(n_components=2)
# x_pca=pca.fit_transform(x_dimred)

# print(f"original shape: {x_dimred.shape}")
# print(f"pca reduced shape: {x_pca.shape}")
# print(f"explained variance ratio: {pca.explained_variance_ratio_}")
# print(f"total variance ratio :{sum(pca.explained_variance_ratio_):.4f}")

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, LogisticRegression
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np



print("\n7) Artificial Neural Network (Digits dataset)")
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
ann.fit(X_train_scaled, y_train)
y_pred = ann.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))


print("\n4) Polynomial Regression (synthetic y = x^2 + noise)")
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
print(X)
y = X[:, 0] ** 2 + np.random.randn(100) * 0.5
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)
print("R^2 score:", r2_score(y_test, y_pred))