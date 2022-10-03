"""
Ideal Model 
Author: cap37
Version: 2.0 Final
Date: 14/09/2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, \
    AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.backend import clear_session
# Good Practice: Free your memory from previously made models.
clear_session()

width = 256
height = 256
img_folder = ""
test_folder = ""

# Training dataset over a 0.2 val split
training_dataset = keras.preprocessing.image_dataset_from_directory(
    img_folder,
    validation_split=0.2, 
    subset='training',
    labels='inferred',
    label_mode='categorical', # one hot encoded for categorical cross entropy loss
    seed=2022,
    shuffle=True, # default option, but good inclusion
    image_size=(height, width),
    batch_size=32,
    color_mode='grayscale' # default is rgb, greyscale = 1 channel, rgb 3 and rgba is 4
)

# Validation dataset of 20%
validation_dataset = keras.preprocessing.image_dataset_from_directory(
    img_folder,
    validation_split=0.2,
    subset='validation',
    labels='inferred',
    label_mode='categorical',
    seed=2022,
    shuffle=True,
    image_size=(height, width),
    batch_size=32,
    color_mode='grayscale'
)

# testing dataset, either synthetic or real
testing_dataset = keras.preprocessing.image_dataset_from_directory(
    test_folder,
    labels='inferred',
    label_mode='categorical',
    seed=2022,
    shuffle=True,
    image_size=(height, width),
    batch_size=1,
    color_mode='grayscale'
)

# Build the Model with the saved hyperparams + Dropout 0.25
model = Sequential()
# gaussian noise is a regulurisation technique that prevents overfitting through random data augmentation
model.add(keras.layers.GaussianNoise(0.2, input_shape=(height,width,1)))

model.add(Conv2D(filters=32, kernel_size=5, input_shape=(height,width,1), 
                 activation='relu', padding='same'))
model.add(AveragePooling2D())
model.add(keras.layers.BatchNormalization())

model.add(Conv2D(filters=20, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=24, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=24, kernel_size=6, activation='relu', padding='same'))
model.add(AveragePooling2D())
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())
# Flatten inputs for Dense layer
model.add(Flatten())

model.add(Dense(5, activation='softmax')) # Last layer: 5 class nodes
model.summary()

from tensorflow.keras.optimizers import Adam

optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_dataset, batch_size=32, epochs=150, validation_data=validation_dataset)

# visualise training results
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"] # on training set
val_loss_values = history_dict["val_loss"] # on validation set
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "--", color='green', label="Training loss")
plt.plot(epochs, val_loss_values, "-", label="Validation loss") 
plt.title("Training and validation loss on greyscale 150 epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("greyscale_loss_results.png")

#plt.clf()

history_dict = history.history
train_acc = history_dict["accuracy"] # on training set
val_acc = history_dict["val_accuracy"] # on validation set
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, train_acc, "--", color='green', label="Training accuracy")
plt.plot(epochs, val_acc, "-", label="Validation accuracy") 
plt.title("Training and validation accuracy on greyscale 150 epochs")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("greyscale_accuracy_results.png")

from sklearn.metrics import classification_report, confusion_matrix

model.evaluate(testing_dataset)
