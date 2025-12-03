#Linear regression
# with built-in function
import numpy as np
from sklearn.linear_model import LinearRegression

# Reshape X for sklearn
X = np.array([1,3,5,2,3]).reshape(-1,1)
Y = np.array([4,1,5,3,2])

model = LinearRegression()
model.fit(X, Y)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Prediction
print("Prediction for X = 6:", model.predict([[6]])[0])

# calculating R^2 statistic
# predicted values
Y_pred = 0.23 * X + 2.34
sum_y = np.sum(Y)
# RSS calculation
rss = np.sum((Y - Y_pred) ** 2)
print("Rss:- ",rss)

RSS = np.sum((Y - Y_pred) ** 2)
TSS = np.sum((Y - sum_y) ** 2)

R2 = 1 - (RSS / TSS)
print("R2:- ",R2)