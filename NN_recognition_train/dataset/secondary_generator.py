import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as k
import glob
from dataset.square_generation import stitch_random_square
from PIL import Image
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
DATASET_DIR = "../data/grass_pretrain"
TRUE_NEGATIVES_DIR = "../data/true_negatives_pretrain"

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

BATCH_SIZE = 96

list_of_grass_images = glob.glob(DATASET_DIR + "/*jpeg")

##########################################
model_to_use = "step1_bigger_longertrain.tf"
##########################################
def custom_mse(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.reduce_mean(tf.math.square(y_true_filtered - y_pred_filtered))
    return loss
model_step_1 = k.models.load_model(f"weights/{model_to_use}", custom_objects = {"custom_mse": custom_mse})


# in here the function should randomly chose to spike in some real data
def generate_batch():
    images = np.empty((BATCH_SIZE,1000,1000,3), dtype = np.float32)
    enc_letter = np.zeros((BATCH_SIZE,1), dtype = np.int32)
    enc_colour = np.zeros((BATCH_SIZE,3), dtype = np.float32)
    i = 0
    with ProcessPoolExecutor(max_workers = 2) as executor:
        for X, coords, letter, colour in executor.map(stitch_random_square, random.sample(list_of_grass_images,BATCH_SIZE)):
            images[i] = X
            enc_letter[i,0] = char_to_int[letter]
            enc_colour[i,:] = np.array(colour)/255
            i = i + 1
    return images, enc_letter, enc_colour

def sanitize(coords):
    A = np.clip(coords[0],0.01,0.99)
    B = np.clip(coords[1],0.01,0.99)
    C = np.clip(coords[2],0.15,0.7)
    D = np.clip(coords[3],0.15,0.7)
    X0 = max(A - D/1.3, 0)
    Y0 = max(B - C/1.3, 0)
    X1 = min(A + D/1.3, 1)
    Y1 = min(B + C/1.3, 1)
    return X0, X1, Y0, Y1

def secondary_generator():
    for i in range(940): # will increase it even more when the step1 is run again with real data spiked in
        X, enc_letter, enc_colour = generate_batch()
        
        presence_pred, coords_pred = model_step_1.predict(tf.image.resize(X.reshape(BATCH_SIZE,1000,1000,3), (224, 224), method="nearest"))
        cropped_images = np.empty((BATCH_SIZE,224,224,3), dtype = np.float32)
        for i in range(BATCH_SIZE):
            X0, X1, Y0, Y1 = sanitize(coords_pred[i])
            img = tf.image.resize(np.expand_dims(X[i, int(Y0*1000):int(Y1*1000)+1, int(X0*1000):int(X1*1000)+1, :], 0),(224,224), method="nearest")
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, 40,)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.image.random_hue(img, 0.05)
            cropped_images[i] = img

        yield cropped_images, (enc_letter, enc_colour)


def retrieve_tf_dataset_secondary(to_cache = True):
    tf_data = tf.data.Dataset.from_generator(secondary_generator, output_types = (tf.float32,(tf.int32, tf.float32)), output_shapes = ((BATCH_SIZE,224,224,3),((BATCH_SIZE,1),(BATCH_SIZE,3))))
    tf_data = tf_data.prefetch(buffer_size = 3)
    if to_cache:
        tf_data = tf_data.cache("/mnt/iusers01/jw01/mdefscs4/scratch/step_2_cache_11-04-2021.tfdata")
    tf_data = tf_data.repeat()
    return tf_data