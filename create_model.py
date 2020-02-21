from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow.keras

import os
import cv2
import numpy as np

#path = "cards_imgs/"
path = "result_photos/"
cards = os.listdir(path=path) 

x_train_cards = []
y_train1 = []

h = 200 # 355
w = 200

for card in cards:    
    image = cv2.imread(path + card, cv2.IMREAD_COLOR)
    x_train_cards.append(image)
    y_train1.append(card.split('[')[1].split(']')[0])

x_train1 = []

for x in x_train_cards:    
    img_g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) # BGR to GRAY
    dim = (h,w)    
    img_rs = cv2.resize(img_g, dim, interpolation=cv2.INTER_AREA) # Изменение размера
    
    x_train1.append(img_rs)

import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import pandas as pd

le = preprocessing.LabelEncoder()
y_train = np.asarray(y_train1)
id_numbers = pd.Series(y_train)

d = {"abbr": y_train, "value": y_train}
df = pd.DataFrame(d["abbr"])

train_lbl = df.apply(le.fit_transform) # Преобразовали слова (действия) в цифры

y_train = to_categorical(train_lbl)

values = y_train1
numbers = np.asarray(train_lbl)
dic = {}
dic_r = {}
i = -1
for n in numbers:
    i += 1
    dic[y_train1[i]] = numbers[i][0]
    dic_r[numbers[i][0]] = y_train1[i]
    """dic.update({
        "value": y_train1[i],
        "number": numbers[i][0]
    })"""
    
print(dic_r)

x_train = x_train1
x_train = np.asarray(x_train)
x_train = x_train.astype('float32')

print(x_train[0].shape)
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], h, w, 1)
print(x_train.shape)


import tensorflow as tf

num_classes = id_numbers.nunique()
batch_size = 24
epochs = 8

model = models.Sequential()
model.add(Conv2D(100, (3, 3), input_shape=(200, 200, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=200, activation='relu'))
#model.add(Dense(units=num_classes, activation='sigmoid'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_train, y_train),
              shuffle=True)

model.save('model.h5')
print("model saved")