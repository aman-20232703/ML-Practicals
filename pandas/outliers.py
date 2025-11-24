import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
data = {'value': [2, 4, 20, 22, 35, 36, 37, 39, 101, 102]}
df = pd.DataFrame(data)
print(df)

Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
print(IQR,"\n")

LB = Q1 - 1.5 * IQR
UB = Q3 + 1.5 * IQR

outliers = df[(df['value'] < LB) | (df['value'] > UB)]
print("\n Identified Outliers using IQR:")
print(outliers)

# using pyplot
plt.figure(figsize=(8, 6))
plt.boxplot(df['value'])
plt.title('Boxplot of Value Column')
plt.xlabel('Categories')
plt.ylabel('Value')
plt.show()

# using seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['value'])
plt.title('Boxplot of Value Column')
plt.ylabel('Value')
plt.xlabel('Categories')
plt.show()