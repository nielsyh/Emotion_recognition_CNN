from __future__ import print_function
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from data_handler import Data
from keras import backend as K
import numpy as np
#import matplotlib.pyplot as plt
data = Data(41904)
x_test, y_test = data.sample_test()
y_test.astype(int)
model=load_model('model_30_adam_default')
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, 50, 50)
else:
    x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements
pred_y= model.predict_classes(x_test)
pred_y.astype(int)
cm= confusion_matrix(y_test, pred_y, [0,1,2,3,4,5,6,7,8,9,10])
#acc = keras.metrics.categorical_accuracy(y_test, pred_y)
print(cm)
#print(str(acc))
#print(accuracy(cm))
class_names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'No-Face']
print(classification_report(y_test, pred_y, target_names=class_names))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
acc = cm.diagonal()
print(acc)