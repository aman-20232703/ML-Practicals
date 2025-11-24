import pandas as pd

# Creating a simple Series
marks = pd.Series([85, 90, 78, 92, 88], 
                  index=["Alice", "Bob", "Charlie", "David", "Esha"])

print("Pandas Series Example:")
print(marks)