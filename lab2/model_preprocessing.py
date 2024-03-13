import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
from sklearn.preprocessing import MinMaxScaler
import os

os.chdir('train')
X_train = np.load('X_train.npy')
os.chdir('../')
os.chdir('test')
X_test = np.load('X_test.npy')

# Все так же код отсюда https://colab.research.google.com/drive/1_Dx-TAsLav17qmOyvc9FlrNZDHDZXP2F?usp=sharing#scrollTo=6OnWEhsqvrUJ
scaler  = MinMaxScaler()
scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

os.chdir('../')
os.chdir('train')
np.save('X_train', X_train)
os.chdir('../')
os.chdir('test')
np.save('X_test', X_test)

