import pandas as pd
import matplotlib as plt

df = pd.read_csv('data/tempData.csv')

print(df.head())
print("======")
print(df.tail())
print("======")
print(df.describe())
print("======")
print(df.info())
