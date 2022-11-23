import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf


training_set = pd.read_csv("train_image.csv")
y_train = training_set['ClassId']
x_train = training_set.iloc[:,0:-1]
test_set = pd.read_csv("test_image.csv")
y_test = test_set.iloc[:,-1]
x_test = test_set.iloc[:,0:-1]

x_train= x_train / 255
x_test= x_test / 255

x_train =tf.convert_to_tensor(x_train)
x_test =tf.convert_to_tensor(x_test)


# Creating NN model
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500, activation="relu", kernel_initializer="HeUniform"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="HeUniform"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(200, activation="relu", kernel_initializer="HeUniform"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="HeUniform"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(43, activation="softmax"))

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20)
