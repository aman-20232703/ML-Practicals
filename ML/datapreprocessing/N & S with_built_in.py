# Normalization with built-in functions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
print(f"Normalized data (scikit-learn): {normalized_data.flatten().tolist()}")

# standardization with built-in functions
import numpy as np
from sklearn.preprocessing import StandardScaler
data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print(f"Standardized data (scikit-learn): {standardized_data.flatten().tolist()}")