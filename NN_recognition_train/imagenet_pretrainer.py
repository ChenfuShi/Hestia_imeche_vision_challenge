import tensorflow as tf
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import random

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

IMAGE_SIZE = 400

name = "test_mobilenet_faster"

def _inverted_res_block(inputs, filters, expansion, stride,):
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
    if stride == 1:
        x = tf.keras.layers.Add()([inputs, x])
        return x
    else:
        return x

def model_to_train():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(6, kernel_size=3, padding='same', activation=None, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = _inverted_res_block(x, filters=10, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=20, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=20, expansion=3, stride=1,)
    x = _inverted_res_block(x, filters=80, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=80, expansion=3, stride=1,)
    x = _inverted_res_block(x, filters=80, expansion=3, stride=1,)
    x = _inverted_res_block(x, filters=40, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=40, expansion=3, stride=1,)
    x = _inverted_res_block(x, filters=40, expansion=3, stride=1,)
    x = _inverted_res_block(x, filters=6, expansion=3, stride=2,)
    x = _inverted_res_block(x, filters=6, expansion=3, stride=1,)
    # x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x) # because last block ends with a batchnorm
    x = tf.keras.layers.ReLU(6.)(x)

    model = tf.keras.Model(inputs, x)

    return model

def augment_tf_dataset(tf_data):
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_contrast(image, 0.8, 1.2), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_brightness(image, 40,), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_saturation(image, 0.8, 1.2), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_hue(image, 0.05), Y)), num_parallel_calls = 6)
    return tf_data

imagenet_localization_train_ds = tf.keras.preprocessing.image_dataset_from_directory("/mnt/iusers01/jw01/mdefscs4/localscratch/imagenet/train/",
                                                                              batch_size=32, image_size=(IMAGE_SIZE,IMAGE_SIZE),seed=123,
                                                                              validation_split=0.05, subset="training",)
imagenet_localization_val_ds = tf.keras.preprocessing.image_dataset_from_directory("/mnt/iusers01/jw01/mdefscs4/localscratch/imagenet/train/",
                                                                              batch_size=32, image_size=(IMAGE_SIZE,IMAGE_SIZE),seed=123,
                                                                              validation_split=0.05, subset="validation",)

class_names = imagenet_localization_train_ds.class_names

# imagenet_localization_train_ds = imagenet_localization_train_ds.cache("/mnt/iusers01/jw01/mdefscs4/localscratch/imagenet_cache.tfdata")
imagenet_localization_train_ds = augment_tf_dataset(imagenet_localization_train_ds)
imagenet_localization_train_ds = imagenet_localization_train_ds.prefetch(buffer_size=200)
imagenet_localization_val_ds = imagenet_localization_val_ds.prefetch(buffer_size=200)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = model_to_train()

inputs = tf.keras.Input(shape=(400, 400, 3))
x = preprocess_input(inputs)
x = base_model(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1000, activation = None, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-3,momentum=0.999)(x)
x = tf.keras.layers.ReLU(6.)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1000, activation = "sigmoid", use_bias=True)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy()])

model.fit(imagenet_localization_train_ds, validation_data = imagenet_localization_val_ds,
    epochs = 5, verbose = 2,)

model.save(f'weights/{name}.tf', save_format = "tf")