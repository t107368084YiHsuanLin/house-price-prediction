# house-price-prediction
first HW of machine learning
# 工作環境
1.Ubuntu 16.04

2.Python 3.5.6

3.Tensorflow 1.10.0

4.Keras 2.2.3

## 作業要求
使用回歸模型做房價預測

1.用train.csv跟valid.csv訓練模型

2.將test.csv中的每一筆房屋參數，輸入訓練好的模型，預測其房價將預測結果上傳

3.看系統幫你算出來的Mean Abslute Error分數夠不夠好？

4.嘗試改進預測模型


## 程式碼

 1.載入需要的數據庫
 
import pandas as pd 

import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import *

from sklearn.preprocessing import *

2.讀取檔案

train   = pd.read_csv('/home/t107368084/machine/train-v3.csv') #載入訓練集

X_train = train.drop(['price','id'],axis=1).values             #因為是訓練資料組所以去除id跟price

Y_train = train['price'].values                                #train裡面的price


valid   = pd.read_csv('/home/t107368084/machine/valid-v3.csv')  #載入驗證集

X_valid = valid.drop(['price','id'],axis=1).values

Y_valid = valid['price'].values


test   = pd.read_csv('/home/t107368084/machine/test-v3.csv')    #載入測試集

X_test = test.drop('id',axis=1).values

3.測試檔案是否讀取成功
  train.shape
  
  valid.shape
  
  test.shape
