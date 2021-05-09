import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random


def retrieve_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation = "relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    presence = tf.keras.layers.Dense(1, activation = "sigmoid", name = "presence")(x)
    coordinates = tf.keras.layers.Dense(8, name = "coordinates")(x)
    letter = tf.keras.layers.Dense(36, activation = "sigmoid", name = "letter")(x)
    model = tf.keras.Model(inputs, [presence, coordinates, letter])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
                loss={"presence":tf.keras.losses.BinaryCrossentropy(from_logits=False), "coordinates":tf.keras.losses.MeanAbsoluteError(), "letter":tf.keras.losses.CategoricalCrossentropy(from_logits=False)},
                metrics={"presence":'accuracy',"coordinates":["mae", "mse"], "letter":"accuracy"})

    return model