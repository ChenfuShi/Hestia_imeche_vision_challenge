import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

def custom_crossentropy(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.keras.losses.categorical_crossentropy(y_true_filtered,y_pred_filtered)
    return loss

def custom_mae(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.reduce_mean(tf.abs(y_true_filtered - y_pred_filtered))
    return loss

def retrieve_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation = "relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    presence = tf.keras.layers.Dense(1, activation = "sigmoid", name = "presence")(x)
    coordinates = tf.keras.layers.Dense(8, name = "coordinates")(x)
    letter = tf.keras.layers.Dense(36, activation = "sigmoid", name = "letter")(x)
    model = tf.keras.Model(inputs, [presence, coordinates, letter])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
                loss={"presence":tf.keras.losses.binary_crossentropy, "coordinates":custom_mae, "letter":custom_crossentropy},
                metrics={"presence":'accuracy'})

    return model