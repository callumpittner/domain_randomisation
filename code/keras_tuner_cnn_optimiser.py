'''
Data Pre-processing and CNN architecture Hyper Tuning with Keras Tuner
Author: cap37
Version: 2.0 Final
Date: 14/09/2022
'''

import os
import keras_tuner
import numpy
from keras import layers
from tensorflow import keras
import tensorflow as tf

# Directories for processing
parameter_name = '' # the test name e.g. rgb Fully Randomised
train_directory = '' # the image training folder
tuner_out_directory = parameter_name + '_tuner_dir'
model_out_filename = parameter_name + '_model.h5'
log_filename = parameter_name + '_training_log'
test_folder = '' # test data filepath

# Processing parameters
image_size = ( 256, 256 )
batch_size = 32
number_of_classes = 5

# Training dataset over a 0.2 val split
train_dataset = keras.preprocessing.image_dataset_from_directory(
    train_directory,
    validation_split=0.2,
    subset='training',
    labels='inferred',
    label_mode='categorical',
    seed=2022,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale' #default is rgb, greyscale = 1 channel, rgb 3 and rgba is 4
)

# Validation dataset of 20%
validation_dataset = keras.preprocessing.image_dataset_from_directory(
    train_directory,
    validation_split=0.2,
    subset='validation',
    labels='inferred',
    label_mode='categorical',
    seed=2022,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

# Builds a model for keras_tuner,
# Adapted from the code @ https://keras.io/guides/keras_tuner/custom_tuner/
# @misc{omalley2019kerastuner,
#     title        = {KerasTuner},
#     author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin,
#     Haifeng and Invernizzi, Luca and others},
#     year         = 2019,
#     howpublished = {\url{https://github.com/keras-team/keras-tuner}}
# }
def build_model(hp):
    """Builds a convolutional model."""
    # input shape, last unit is depth, 1 for greyscale, 3 for rgb
    inputs = keras.Input(shape=(256, 256, 1))
    # assign reuse variable as per keras_tuner documentation
    x = inputs
    # add a gaussian noise layer over input data for regularisation purposes
    x = keras.layers.GaussianNoise( 0.2 )( x )

    # loop over a choice of layers between 2-7
    for i in range(hp.Int('conv_layers', 2, 7, default=5)):
        # choose filters, kernel_sizes, activation and paddings for each layer in loop
        x = keras.layers.Conv2D(
            filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 7),
            activation='relu',
            padding='same')(x)

        # add average or max pooling layers after convolutional layer
        if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
            x = keras.layers.MaxPooling2D()(x)
        else:
            x = keras.layers.AveragePooling2D()(x)

        # apply batch normalisation before activation function as per literature review.
        x = keras.layers.BatchNormalization()(x)
        # activation function
        x = keras.layers.ReLU()(x)

    # add global pooling, max or average
    if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense( number_of_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # assign model optimisers and losses
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Perform keras_tuner search over build_model function
tuner = keras_tuner.RandomSearch(
    #hypermodel
     build_model,
     objective='val_accuracy',
     max_trials=25,
     directory=tuner_out_directory,
     project_name='keras_tuner_results',
     overwrite=True
)

# search over 60 epochs with early stopping
tuner.search(train_dataset,
             epochs=60,
             validation_data=validation_dataset,
             callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=12)]
)


best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
model = tuner.hypermodel.build(best_hps)

# create a history logger to see training process over runtime, metrics are written here.
history_logger = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=",",
    append=True
)

# train the model
history = model.fit(
      train_dataset
    , epochs = 150
    , validation_data = validation_dataset
    , callbacks=[ history_logger ]
)

print(model.summary())

# visualise the loss values over epochs
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "--", color='green', label="Training loss")
plt.plot(epochs, val_loss_values, "-", label="Validation loss") 
plt.title("Train/Val loss Greyscale 150 epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("rgb_loss_results.png")

plt.clf()

# visualise the accuracy values over epochs
history_dict = history.history
train_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, train_acc, "--", color='green', label="Training accuracy")
plt.plot(epochs, val_acc, "-", label="Validation accuracy") 
plt.title("Training and Val accuracy on Greyscale 150 epochs")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("rgb_accuracy_results.png")

# save the model
model.save(model_out_filename)
