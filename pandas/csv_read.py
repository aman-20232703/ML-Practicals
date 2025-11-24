import csv
# without pandas
with open('customers-100.csv', mode='r') as file:
    lines=file.readlines()
    for line in lines:
        data=line.strip().split(",")
        print(data)   # %pip install seaborn
        
# with pandas
import pandas as pd
df= pd.read_csv('customers-100.csv')
# print(df)
# basics pandas operations
print("all information: ",df.info())
print("shape of file: ",df.shape)
print("number of columns: ",df.columns)
print("first 5 value: ",df.head())
print("last 5 value: ",df.tail())
print("all information: ",df.describe())
print("by positon:",df.iloc(1))
print("by vlaue: ",df.loc(1))

