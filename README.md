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

import pandas as pd                           #載入需要的數據庫
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import *
