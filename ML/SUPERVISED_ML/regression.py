import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

# Load Diabetes dataset (regression)
data_reg = load_diabetes()
X_reg = data_reg.data
y_reg = data_reg.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 2.1 Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
y_pred_lin = lin_reg.predict(X_test_r)
lin_r2 = r2_score(y_test_r, y_pred_lin)
lin_mse = mean_squared_error(y_test_r, y_pred_lin)
print(f"Linear Regression R² Score: {lin_r2:.4f}")
print(f"Linear Regression MSE: {lin_mse:.4f}")

# 2.2 Ridge Regression
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_r, y_train_r)
y_pred_ridge = ridge_reg.predict(X_test_r)
ridge_r2 = r2_score(y_test_r, y_pred_ridge)
print(f"Ridge Regression R² Score: {ridge_r2:.4f}")

# 2.3 Lasso Regression
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train_r, y_train_r)
y_pred_lasso = lasso_reg.predict(X_test_r)
lasso_r2 = r2_score(y_test_r, y_pred_lasso)
print(f"Lasso Regression R² Score: {lasso_r2:.4f}")

# 2.4 Support Vector Regression (SVR)
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_r, y_train_r)
y_pred_svr = svr_reg.predict(X_test_r)
svr_r2 = r2_score(y_test_r, y_pred_svr)
print(f"SVR R² Score: {svr_r2:.4f}")





