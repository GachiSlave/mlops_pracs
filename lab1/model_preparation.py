from sklearn.linear_model import LogisticRegression
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего\
import pickle
import os

os.chdir('train')
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Т.к таргет бинарый признаки 1 или 0, использую Логистическую регрессию
# Код отсюда https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = LogisticRegression(random_state=42).fit(X_train, y_train)

pkl_filename = "pickle_model.pkl"

# Код отсюда https://rukovodstvo.net/posts/id_1322/
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

#print(model.score(X_train, y_train))