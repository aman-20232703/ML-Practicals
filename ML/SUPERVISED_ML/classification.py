import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

# Load Breast Cancer dataset (binary classification)
data_class = load_breast_cancer()
X_class = data_class.data
y_class = data_class.target


# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

# 1.1 Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, max_iter=1000 )
log_reg.fit(X_train_c, y_train_c)
log_accuracy = log_reg.score(X_test_c, y_test_c)
print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")

# 1.2 Support Vector Machine (SVM)
from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_c, y_train_c)  # aman
svm_accuracy = svm_clf.score(X_test_c, y_test_c)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# 1.3 Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train_c, y_train_c)
dt_accuracy = dt_clf.score(X_test_c, y_test_c)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# 1.4 Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train_c, y_train_c)
nb_accuracy = nb_clf.score(X_test_c, y_test_c)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

# 1.5 k-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_c, y_train_c)
knn_accuracy = knn_clf.score(X_test_c, y_test_c)
print(f"KNN Accuracy: {knn_accuracy:.4f}")

