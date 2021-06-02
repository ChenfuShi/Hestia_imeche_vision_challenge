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


list_of_grass_images = glob.glob(DATASET_DIR + "/*jpeg")
list_of_negative_images = glob.glob(TRUE_NEGATIVES_DIR + "/*jpeg")

model_to_use = "step1_without_trainable.tf"
def custom_mse(y_true,y_pred):
    y_pred_filtered = y_pred[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    y_true_filtered = y_true[~tf.math.is_nan(tf.reduce_sum(y_true,axis = 1))]
    loss = tf.reduce_mean(tf.math.square(y_true_filtered - y_pred_filtered))
    return loss
model_step_1 = k.models.load_model(f"weights/{model_to_use}", custom_objects = {"custom_mse": custom_mse})



def generate_batch():
    images = np.empty((32,1000,1000,3), dtype = np.float32)
    enc_letter = np.zeros((32,36), dtype = np.float32)
    enc_colour = np.zeros((32,3), dtype = np.float32)
    i = 0
    with ProcessPoolExecutor(max_workers = 8) as executor:
        for X, coords, letter, colour in executor.map(stitch_random_square, random.sample(list_of_grass_images,32)):
            images[i] = X
            enc_letter[i, char_to_int[letter]] = 1
            enc_colour[i] = np.array(colour)/255
            i = i + 1
    return images, enc_letter, enc_colour

def sanitize(coords):
    A = np.clip(coords[0],0.01,0.99)
    B = np.clip(coords[1],0.01,0.99)
    C = np.clip(coords[2],0.1,0.7)
    D = np.clip(coords[3],0.1,0.7)
    X0 = max(A - D/1, 0)
    Y0 = max(B - C/1, 0)
    X1 = min(A + D/1, 1)
    Y1 = min(B + C/1, 1)
    return X0, X1, Y0, Y1

def secondary_generator():
    while True:
        X, enc_letter, enc_colour = generate_batch()
        
        presence_pred, coords_pred = model_step_1.predict(tf.image.resize(X.reshape(32,1000,1000,3), (224, 224)))
        cropped_images = np.empty((32,224,224,3), dtype = np.float32)
        for i in range(32):
            X0, X1, Y0, Y1 = sanitize(coords_pred[i])
            img = tf.image.resize(np.expand_dims(X[i, int(Y0*1000):int(Y1*1000)+1, int(X0*1000):int(X1*1000)+1, :], 0),(224,224))
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, 40,)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.image.random_hue(img, 0.05)
            cropped_images[i] = img

        yield cropped_images, (enc_letter, enc_colour)


def retrieve_tf_dataset_secondary():
    tf_data = tf.data.Dataset.from_generator(secondary_generator, output_types = (tf.float32,(tf.float32, tf.float32)), output_shapes = ((32,224,224,3),((32,36),(32,3))))
    tf_data = tf_data.prefetch(buffer_size = 3)
    return tf_data