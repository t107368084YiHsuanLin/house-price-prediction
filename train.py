import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import *


data_1 = pd.read_csv('/home/t107368084/machine/train-v3.csv')
X_train = data_1.drop(['price','id'],axis=1).values
Y_train = data_1['price'].values

data_2 = pd.read_csv('/home/t107368084/machine/valid-v3.csv')
X_valid = data_2.drop(['price','id'],axis=1).values
Y_valid = data_2['price'].values

data_3 = pd.read_csv('/home/t107368084/machine/test-v3.csv')
X_test = data_3.drop('id',axis=1).values


X_train=scale(X_train)
X_valid=scale(X_valid)
X_test=scale(X_test)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(32, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(X_train.shape[1], input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(1,  kernel_initializer='normal'))

model.compile(loss='MAE', optimizer='adam')

epochs = 100
batch_size = 64

file_name=str(epochs)+'_'+str(batch_size)
TB=TensorBoard(log_dir='logs/'+file_name, histogram_freq=0)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(X_valid, Y_valid),callbacks=[TB])
model.save("/home/t107368084/machine/model.h5")

Y_predict = model.predict(X_test)
np.savetxt('/home/t107368084/machine/test.csv', Y_predict, delimiter = ',')