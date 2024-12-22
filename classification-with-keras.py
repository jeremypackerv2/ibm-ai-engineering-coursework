# Author: Jeremy Packer
# Date: December 22, 2024
# Description: Classification of MNIST( Modified National Institute of Standards and Technology) with Keras to distinguish characters. 
# The MNIST database contains 60,000 training images and 10,000 testing images of digits written by high school students and employees of the United States Census Bureau.
 

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# confirm machine handles this venv properly
print("Numpy version:", np.__version__)     
print("Pandas version:", pd.__version__)
print("TensorFlow version:", tf.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)

# load the Modified National Institute of Standards and Technology database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

plt.imshow(X_train[0])

# flatten images into one-dimensional vectors
num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

# build a neural network for classification
# define classification model
def classification_model():
    # create model
    
    model = Sequential()
    model.add(Input(shape=(num_pixels,)))
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0) 

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# save the model if I want to
model.save('classification_model.keras')

# load the model if I want to for further processing
model = tf.keras.models.load_model('classification_model.keras')
