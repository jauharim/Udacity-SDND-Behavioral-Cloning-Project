import csv
import cv2
import numpy as np
#%tensorflow_version 1.x 
#import tensorflow as tf
import keras

lines=[]
    
with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]

for line in lines:
  for i in range(3):
    source_path=line[i]
    tokens=source_path.split('/')
    filename=tokens[-1]
    local_path='./data/IMG/'+filename
    #print(local_path)
    image=cv2.imread(local_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #print(image)
    #exit()
    images.append(image)
  correction=0.2
  measurement=float(line[3])
  measurements.append(measurement)
  measurements.append(measurement+correction)
  measurements.append(measurement-correction)

X_train=np.array(images)
y_train=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add((Cropping2D(cropping=((70,25),(0,0)))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)

model.save('model.h5')

