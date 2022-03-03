# -*- coding: utf-8 -*-
# Created on Tue Oct  5 15:51:06 2021

# import CNN libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# initialise the model
classifier = Sequential()

# convolutional layer
bit_depth = 3 # for color images (r/g/b)
# bit_depth = 1 # for monochrome images

# convolution2D parameters: 
# 1) number of filters
# 2) stride size
# 3) image size
classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,bit_depth), activation='relu'))

# maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# flatten
classifier.add(Flatten())

# fully connected layer
# hidden layer 1 : nodes=8
classifier.add(Dense(units=pow(2,3),activation='relu'))
# hidden layer 2 : nodes=4
classifier.add(Dense(units=pow(2,2),activation='relu'))
# output layer : nodel=1
classifier.add(Dense(units=1,activation='sigmoid'))

# compile the model
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# image pre-processing and image augmentation

from keras.preprocessing.image import ImageDataGenerator

# apply the augmentation on train and validation images
train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range=0.3,
                                   zoom_range=0.4,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range=0.3,
                                   zoom_range=0.4,
                                   horizontal_flip=True)

# set up the directory structure to read the images
train_dir="C:\\Users\\Sahil\\Desktop\\Imarticus Learning -4\\DAY 65 Image Cat Dog Data CNN\\binary\\train"
val_dir="C:\\Users\\Sahil\\Desktop\\Imarticus Learning -4\\DAY 65 Image Cat Dog Data CNN\\binary\\validation"

# read the images
train_set = train_datagen.flow_from_directory(train_dir,
                                              target_size=(64,64),
                                              # color_mode='grayscale',
                                              batch_size=5,
                                              class_mode='binary')

val_set = val_datagen.flow_from_directory(val_dir,
                                          target_size=(64,64),
                                          batch_size=5,
                                          class_mode='binary' )

# train the model
# start with a small number for epochs
EPOCHS = 15

# fit for all images 
# steps_per_epoch = train_set.n
# validation_steps = val_set.n
classifier.fit(train_set,steps_per_epoch = 25,
               epochs = EPOCHS,
               validation_data = val_set,
               validation_steps = 25)

# pip install pillow

# predict the test images
from keras.preprocessing import image
import numpy as np
import os

# test path
test_dir = "C:\\Users\\Sahil\\Desktop\\Imarticus Learning -4\\DAY 65 Image Cat Dog Data CNN\\binary\\test\\"

# create a list to store the file names of the test images
testimages = []

for p,d,files in os.walk(test_dir):
    for f in files:
        testimages.append(p+f)
        
print(testimages)

# stack up the images for prediction
imagestack = None

for i in testimages:
    img = image.load_img(i,target_size=(64,64))
    
    # convert image to an array format
    y = image.img_to_array(img)
    y = np.expand_dims(y,axis=0)
    y /= 255 # rescale image to match the Image Generator setting
    
    if imagestack is None:
        imagestack = y
    else:
        imagestack = np.vstack([imagestack,y])

print(imagestack)    
len(imagestack)

# predict
predy = classifier.predict_classes(imagestack)
predy = predy.reshape(-1)
predy

# try this
classifier.predict(imagestack)

# cat 0
# dog 1
