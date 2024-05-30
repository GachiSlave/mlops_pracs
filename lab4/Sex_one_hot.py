import pandas as pd

df = pd.read_csv('titanic.csv')

one_hot = pd.get_dummies(df['Sex']).astype('uint8')


df = df.drop(columns = ['Sex'])

df = df.join(one_hot)

print(df['male'])

df = df.drop(columns = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])

print(df.columns)

df.to_csv('titanic.csv')