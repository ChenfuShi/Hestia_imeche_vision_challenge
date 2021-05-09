import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as k
import glob
from dataset.square_generation import stitch_random_square
from PIL import Image

DATASET_DIR = "../data/grass_pretrain"
TRUE_NEGATIVES_DIR = "../data/true_negatives_pretrain"


STITCH_PROB = 0.5
TRUE_NEG_PRO = 0.2

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


list_of_grass_images = glob.glob(DATASET_DIR + "/*jpeg")
list_of_negative_images = glob.glob(TRUE_NEGATIVES_DIR + "/*jpeg")

def train_generator():
    for i in range(10000):
        if random.random() > TRUE_NEG_PRO:
            if random.random() > STITCH_PROB:
                X, coords, letter, color = stitch_random_square(random.choice(list_of_grass_images))
                presence = 1
                position = [coords["A_X"]/1000,coords["B_X"]/1000,coords["C_X"]/1000,coords["D_X"]/1000,coords["A_Y"]/1000,coords["B_Y"]/1000,coords["C_Y"]/1000,coords["D_Y"]/1000]
                enc_letter = np.zeros(36)
                enc_letter[char_to_int[letter]] = 1
            else:
                img_file = random.choice(list_of_grass_images)
                X = np.array(Image.open(img_file))
                presence = 0
                position = [None,None,None,None,None,None,None,None,]
                enc_letter = np.full(36, None)
        else:
            img_file = random.choice(list_of_negative_images)
            X = np.array(Image.open(img_file))
            presence = 0
            position = [None,None,None,None,None,None,None,None,]
            enc_letter = np.full(36, None)
        # preprocess input for imagenet style
        yield X, (presence, position, enc_letter)

def retrieve_tf_dataset():
    tf_data = tf.data.Dataset.from_generator(train_generator, output_types = (tf.float32,(tf.float32,tf.float32,tf.float32)), output_shapes = ((1000,1000,3),((),(8),(36))))

    tf_data = tf_data.prefetch(buffer_size = 200)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.resize(image, (224, 224)), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_contrast(image, 0.8, 1.2), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_brightness(image, 40,), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_saturation(image, 0.8, 1.2), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_hue(image, 0.05), Y)))

    tf_data = tf_data.batch(32)
    return tf_data