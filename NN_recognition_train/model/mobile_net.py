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
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1000)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_1 = tf.keras.layers.Dense(1)(x)
    output_2 = tf.keras.layers.Dense(8)(x)
    output_3 = tf.keras.layers.Dense(36)(x)
    model = tf.keras.Model(inputs, [output_1, output_2, output_3])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003),
                loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.CategoricalCrossentropy(from_logits=True)],
                metrics=['accuracy',"mae", "mse"])

    return model