import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

from dataset.secondary_generator import retrieve_tf_dataset_secondary, secondary_generator

IMAGE_SIZE = 224

def retrieve_mobilenet_model():
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = preprocess_input(inputs)
    x = tf.keras.applications.MobileNetV2(include_top = False, alpha = 0.5, input_shape = (224,224,3), weights = "imagenet")(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000, activation = None, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    letter = tf.keras.layers.Dense(36, activation = "sigmoid", name = "letter")(x)
    colour = tf.keras.layers.Dense(3, name = "colour")(x)
    model = tf.keras.Model(inputs, [letter, colour])
    return model

def train_this(model_name):

    tf_data = retrieve_tf_dataset_secondary()
    # for X,Y in tf_data.take(2050):
    #     pass
    model = retrieve_mobilenet_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss={"letter":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), "colour":"mae"},
                metrics={"letter":"accuracy","colour":["mae","mse"]},
                loss_weights={"letter":1, "colour":1})
    model.summary()            
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(f'weights/{model_name}_epoch_5.tf', period=5) 
    print(model_name)
    model.fit(tf_data, epochs = 500, verbose = 2, steps_per_epoch = 100,)# callbacks=[checkpoint])
    
    model.save(f'weights/{model_name}.tf', save_format = "tf")

    model.evaluate(tf_data, steps = 50, verbose = 2)