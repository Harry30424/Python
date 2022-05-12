# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:30:13 2022

@author: user
"""






# tf.__version__
#import tensorflow_hub as hub


# from keras.models import Sequential
from tensorflow import keras
# from keras import layers
# from keras.layers.convolutional import Convolution2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from skimage import io
from skimage.transform import resize
from keras import backend
import os
import cv2
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#load datasets
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0
print(len(x_train)) #6000 img
print(x_train[0].shape) #each img is 28x28

#build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #accurancy 對應graph

#將搭好的model 去fit our training data
history = model.fit(x_train, y_train, epochs=5,validation_split=0.2, batchsize=200)

#evaluate model 打分數
model.evaluate(x_test,y_test,verbose=2)

#draw  graph to present accurancy
import matplotlib.pyplot as plt

#draw accuracy of model
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'g')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accurancy')
plt.legend(['train','val'], loc='upper left')
plt.show()

print('train_acc:', history.history['accuracy'][-1])
print('val_acc:', history.history['val_accuracy'][-1])

#draw loss function of model
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'g')
plt.title('model loss')
plt.xlabel('loss') #numbers of training
plt.ylabel('epoch')
plt.legend(['train','val'], loc='lower right') #change location of label
plt.show()

print('train_loss:', history.history['loss'][-1])
print('val_loss:', history.history['val_loss'][-1])

#pred 20筆
pred = model.predict(x_test)
print("prediction:", np.argmax(pred[:20],axis=1))
print("actual:    ",y_test[:20])
len(pred)

#show error graph
x1 = x_test[6,:,:]
plt.imshow(x1.reshape(28,28))
plt.show()

#test img

file= 'testimg.png'
image = io.imread(file, as_gray=True)
image_resized = resize(image,(28,28),anti_aliasing=True)
x2 = image_resized.reshape(1,28,28)
x2 = np.abs(1-x2)
pred2 = model.predict(x2)
print(np.argmax(pred2,axis=1)) #print prediction result

#save model
#save to tensorflow->m
save = "./tensorflow/menist_model/"
path = os.path.join(save,'keras_mnist1.h5') #path+modelname
model.save(path)



# fig = plt.figure()
# for i in range(15):
#     plt.subplot(5, 5, i+1)   # 用3行5列形式展示
#     plt.imshow(x_train[i], cmap = 'Grays')     # 用灰色顯示圖像灰度值
#     plt.xticks([])           # 刪除x軸標記，否則會自動錶上座標
#     plt.yticks([])

    
# def plot_images_labels_prediction(images,labels,prediction,idx,num=10): 
  
#   # 設定顯示圖形的大小
#   fig= plt.gcf()
#   fig.set_size_inches(12,14)

#   # 最多25張
#   if num>25:num=25

#   # 一張一張畫
#   for i in range(0,num):

#     # 建立子圖形5*5(五行五列)
#     ax=plt.subplot(5,5,i+1)

#     # 畫出子圖形
#     ax.imshow(images[idx],cmap='binary')

#     # 標題和label
#     title="label=" +str(labels[idx])

#     # 如果有傳入預測結果也顯示
#     if len(prediction)>0:
#       title+=",predict="+str(prediction[idx])

#     # 設定子圖形的標題大小
#     ax.set_title(title,fontsize=10)

#     # 設定不顯示刻度
#     ax.set_xticks([]);ax.set_yticks([])  
#     idx+=1
#   plt.show()

# plot_images_labels_prediction(x_test,y_train,pred,idx=340)





#clear model
keras.backend.clear_session()
#clear matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"







