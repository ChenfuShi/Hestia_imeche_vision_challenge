import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

from dataset.secondary_generator import retrieve_tf_dataset_secondary, secondary_generator

IMAGE_SIZE = 224

def _inverted_res_block(inputs, filters, expansion, stride, add = True):
    x = inputs
    # expand
    x = tf.keras.layers.Conv2D(filters * expansion, kernel_size=1, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # depthwise conv
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides = stride, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # project
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    if stride == 1 and add:
        x = tf.keras.layers.Add()([inputs, x])
        return x
    else:
        return x

def model_to_train():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation=None, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = _inverted_res_block(x, filters=20, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=30, expansion=6, stride=2,)
    x = _inverted_res_block(x, filters=30, expansion=6, stride=1,)
    x = _inverted_res_block(x, filters=40, expansion=6, stride=2,)
    x = _inverted_res_block(x, filters=60, expansion=6, stride=1, add = False)
    x = _inverted_res_block(x, filters=80, expansion=6, stride=1, add = False)
    x = _inverted_res_block(x, filters=80, expansion=6, stride=1)
    x = tf.keras.layers.Conv2D(100, kernel_size=3, padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)

    model = tf.keras.Model(inputs, x)

    return model

def retrieve_mobilenet_model():
    base_model = model_to_train()
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(500, activation = None, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    letter = tf.keras.layers.Dense(36, activation = "sigmoid", name = "letter")(x)
    colour = tf.keras.layers.Dense(3, name = "colour")(x)
    model = tf.keras.Model(inputs, [letter, colour])
    return model

def train_this(model_name):

    tf_data = retrieve_tf_dataset_secondary()

    model = retrieve_mobilenet_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                loss={"letter":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), "colour":"mse"},
                metrics={"letter":"accuracy","colour":["mae","mse"]})
                
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'weights/{model_name}_epoch_5.tf', period=5) 
    model.fit(tf_data, epochs = 10, verbose = 2, steps_per_epoch = 50, callbacks=[checkpoint])
    
    model.save(f'weights/{model_name}.tf', save_format = "tf")

    model.evaluate(tf_data, steps = 10, verbose = 2)