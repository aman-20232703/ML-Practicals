# Normalization without built-in functions
import pandas as pd

def min_max_normalize(data):
    min_x = min(data)
    max_x = max(data)
    normalized_data = [(x - min_x) / (max_x - min_x) for x in data]
    return normalized_data

# Example
data = [10, 20, 30, 40, 50]
normalized_data = min_max_normalize(data)
print(f"Original data: {data}")
print(f"Normalized data (without built-in function): {normalized_data}")

print("\n")

# standardization without built-in functions
def z_score_standardize(n):
    mean = sum(n) / len(n)
    variance = sum([(x - mean) ** 2 for x in n]) / len(n)
    std_dev = variance ** 0.5
    standardized_data = [(x - mean) / std_dev for x in data]
    return standardized_data

# Example:
n = [10, 20, 30, 40, 50]
standardized_data = z_score_standardize(n)
print(f"Original data: {n}")
print(f"Standardized data (without built-in function): {standardized_data}")