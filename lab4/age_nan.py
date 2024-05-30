import pandas as pd

df = pd.read_csv('titanic.csv')

df['Age'] = df['Age'].fillna(value=df['Age'].mean())

print(df['Age'])

df.to_csv('titanic.csv')