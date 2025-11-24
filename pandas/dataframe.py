import pandas as pd

# Creating a DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Esha"],
    "Age": [23, 24, 22, 25, 23],
    "Marks": [85, 90, 78, 92, 88]
}

students_df = pd.DataFrame(data)

print("Pandas DataFrame Example:")
print(students_df)
