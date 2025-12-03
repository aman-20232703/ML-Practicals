# without built-in function
import numpy as np

# Data
X = np.array([1,3,5,2,3])
Y = np.array([4,1,5,3,2])
n = len(X)

# Calculate slope (m)
sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X * X)

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

# Calculate intercept (c)
c = (sum_y - m * sum_x) / n

print("Manual Slope (m):", m)
print("Manual Intercept (c):", c)

# Predicting Y for X = 6
X_new = 6
Y_pred = m * X_new + c
print("Prediction for X = 6:", Y_pred)

# calculating R^2 statistic
# predicted values
Y_pred = 0.23 * X + 2.34

# RSS calculation
rss = np.sum((Y - Y_pred) ** 2)
print("Rss:- ",rss)

RSS = np.sum((Y - Y_pred) ** 2)
TSS = np.sum((Y - sum_y) ** 2)

R2 = 1 - (RSS / TSS)
print("R2:- ",R2)