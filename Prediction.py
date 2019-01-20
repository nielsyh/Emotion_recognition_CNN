from __future__ import print_function
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from data_handler import Data
from keras import backend as K
import matplotlib.pyplot as plt
data = Data(41904)
x_test = data.test_x
y_test = data.test_y
y_test.astype(int)
model=load_model('/Users/Chiara/Desktop/Emotion_recognition_CNN/models/Tanh_WeightNormal_Batch_L2.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, 50, 50)

else:
    x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)

pred_y1= model.predict_classes(x_test)
pred_y1.astype(int)
cm= confusion_matrix(y_test, pred_y1, [0,1,2,3,4,5,6,7,8,9,10])
print(cm)
print('Predicted1:', pred_y1)
print('True:', y_test)
