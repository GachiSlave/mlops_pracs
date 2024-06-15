import os
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего\
import pickle



pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Теперь тестовую пихаю
os.chdir('test')
X_test= np.load('X_test.npy')
y_test = np.load('y_test.npy')

score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
