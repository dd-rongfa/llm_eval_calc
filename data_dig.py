import pandas as pd


fp = './data/dsr1_ark/dsr1_ark_details.csv'

df = pd.read_csv(fp)

print(df.head())
# row count
print(len(df))

# column count
print(len(df.columns))

# column names
print(df.columns)



