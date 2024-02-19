import pandas as pd # Библиотека Pandas для работы с табличными данными
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
from sklearn.model_selection import train_test_split
import os

# Это что бы все колоночки отображались
pd.set_option('display.max_columns', None)

url = 'https://raw.githubusercontent.com/DanilaAkh/mlops_pracs/main/lab1/heart.csv'
response = requests.get(url)

with open('data.csv', 'wb') as file:
    file.write(response.content)

# Взял от сюда https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
df = pd.read_csv('data.csv')
# print(df)
# print(df.info())

# Этот участок кода отсюда https://colab.research.google.com/drive/1ZvcWGTGUWbzqhZ1cbNwq4LgztayuScxc?usp=sharing#scrollTo=LTDL0Ip9vrTa
cat_columns = [] # создаем пустой список для имен колонок категориальных данных
num_columns = [] # создаем пустой список для имен колонок числовых данных

for column_name in df.columns: # смотрим на все колонки в датафрейме
    if (df[column_name].dtypes == object): # проверяем тип данных для каждой колонки
        cat_columns +=[column_name] # если тип объект - то складываем в категориальные данные
    else:
        num_columns +=[column_name] # иначе - числовые

# важно: если признак категориальный, но хранится в формате числовых данных, тогда код не сработает корректно


# выводим результат
# print('Категориальные данные:\t ',cat_columns, '\n Число столблцов = ',len(cat_columns))
#
# print('Числовые данные:\t ',  num_columns, '\n Число столблцов = ',len(num_columns))


# Использую только числовые признаки
# Взял участок кода от сюда https://colab.research.google.com/drive/1_Dx-TAsLav17qmOyvc9FlrNZDHDZXP2F?usp=sharing#scrollTo=6OnWEhsqvrUJ
df_num = df[num_columns].copy()

# За таргет взял есть ли болезнь сердца - бинарный признак 0, 1
X,y = df_num.drop(columns = ['HeartDisease']).values, df_num['HeartDisease'].values

features_names = df_num.drop(columns = ['HeartDisease']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Работа с папками
if os.path.exists('test'):
    pass
else:
    os.mkdir('test')

if os.path.exists('train'):
    pass
else:
    os.mkdir('train')

# Читатель может спросить - Влад, зачем ходить по папочкам когда можно было просто написать
# что то типо такого np.save('train//X_train', X_train), а я скажу, после запуска на ВМ линукс, там вообще все не
# Слава богу происходить, поэтому так
os.chdir('train')
np.save('X_train', X_train)
np.save('y_train', y_train)

os.chdir('../')
os.chdir('test')
np.save('X_test', X_test)
np.save('y_test', y_test)
