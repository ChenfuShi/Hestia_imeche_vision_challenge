import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

from dataset.train_generator import retrieve_tf_dataset

def custom_crossentropy(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.keras.losses.categorical_crossentropy(y_true_filtered,y_pred_filtered)
    return loss

def custom_mse(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.reduce_mean(tf.math.square(y_true_filtered - y_pred_filtered))
    return loss

def retrieve_mobilenet_model():
    base_model = tf.keras.models.load_model('weights/test_mobilenet_faster.tf').layers[3]
    base_model.trainable = False
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(400, 400, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation = None, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    presence = tf.keras.layers.Dense(1, activation = "sigmoid", name = "presence")(x)
    coordinates = tf.keras.layers.Dense(4, name = "coordinates")(x)
    letter = tf.keras.layers.Dense(36, activation = "sigmoid", name = "letter")(x)
    model = tf.keras.Model(inputs, [presence, coordinates, letter])
    return model

def train_this(model_name):

    tf_data = retrieve_tf_dataset()

    model = retrieve_mobilenet_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
                loss={"presence":tf.keras.losses.binary_crossentropy, "coordinates":custom_mse, "letter":custom_crossentropy},
                metrics={"presence":"accuracy",})

    model.fit(tf_data, epochs = 3, verbose = 2,)

    model.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005),
                loss={"presence":tf.keras.losses.binary_crossentropy, "coordinates":custom_mse, "letter":custom_crossentropy},
                metrics={"presence":"accuracy",})

    model.fit(tf_data, epochs = 5, verbose = 2,)
    
    model.save(f'weights/{model_name}.tf', save_format = "tf")