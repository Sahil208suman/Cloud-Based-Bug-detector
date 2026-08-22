import pandas as pd

data = pd.read_csv("data/bugs.csv")

print(data.head())
print("\nNumber of rows:", len(data))
print("\nColumns:", data.columns.tolist())
print("\nBug counts:")
print(data["bug"].value_counts())
