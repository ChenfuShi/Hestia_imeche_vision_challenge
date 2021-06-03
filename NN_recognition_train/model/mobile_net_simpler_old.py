import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

from dataset.train_generator import retrieve_tf_dataset

IMAGE_SIZE = 224

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

def _inverted_res_block(inputs, filters, expansion, stride, add = True):
    x = inputs
    # expand
    x = tf.keras.layers.Conv2D(filters * expansion, kernel_size=1, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # depthwise conv
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides = stride, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # project
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and add:
        x = tf.keras.layers.Add()([inputs, x])
        return x
    else:
        return x

def model_to_train(inputs):
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides = 2, padding='same', activation=None, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = _inverted_res_block(x, filters=16, expansion=1, stride=1, add = False)
    x = _inverted_res_block(x, filters=32, expansion=6, stride=2,)
    x = _inverted_res_block(x, filters=32, expansion=6, stride=1,)
    x = _inverted_res_block(x, filters=64, expansion=6, stride=2,)
    x = _inverted_res_block(x, filters=64, expansion=6, stride=1,)
    x = _inverted_res_block(x, filters=32, expansion=6, stride=2,)
    x = _inverted_res_block(x, filters=32, expansion=6, stride=1,)
    x = tf.keras.layers.Conv2D(9, kernel_size=1, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    return x

def retrieve_mobilenet_model():
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = preprocess_input(inputs)
    x = model_to_train(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation = None, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    presence = tf.keras.layers.Dense(1, activation = "sigmoid", name = "presence")(x)
    coordinates = tf.keras.layers.Dense(4, name = "coordinates")(x)
    model = tf.keras.Model(inputs, [presence, coordinates])
    return model

def train_this(model_name):

    tf_data = retrieve_tf_dataset()

    model = retrieve_mobilenet_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                loss={"presence":tf.keras.losses.binary_crossentropy, "coordinates":custom_mse},
                metrics={"presence":"accuracy",})
    model.summary()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'weights/{model_name}_epoch_5.tf', period=5) 
    print(model_name)
    model.fit(tf_data, epochs = 10, verbose = 2, steps_per_epoch = 100, callbacks=[checkpoint])
    
    model.save(f'weights/{model_name}.tf', save_format = "tf")

    model.evaluate(tf_data, steps = 10, verbose = 2)