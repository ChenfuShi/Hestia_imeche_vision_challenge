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

def align_coords(coords):
    sums = np.array((coords["A_X"] + coords["A_Y"], coords["B_X"] + coords["B_Y"], coords["C_X"] + coords["C_Y"], coords["D_X"] + coords["D_Y"]))
    X = np.array((coords["A_X"], coords["B_X"], coords["C_X"], coords["D_X"]))
    Y = np.array((coords["A_Y"], coords["B_Y"], coords["C_Y"], coords["D_Y"]))
    idx_Y = np.argsort(Y)
    top_left = np.argmin(sums)
    bottom_right = np.argmax(sums)
    a = list(idx_Y)
    a.remove(top_left)
    a.remove(bottom_right)
    top_right = a[0]
    bottom_left = [1]
    return np.array((X[top_left]/1000, Y[top_left]/1000, X[bottom_right]/1000, Y[bottom_right]/1000, X[top_right]/1000, Y[top_right]/1000, X[bottom_left].item()/1000, Y[bottom_left].item()/1000,))

def train_generator():
    while True:
        if random.random() > TRUE_NEG_PRO:
            if random.random() > STITCH_PROB:
                X, coords, letter, color = stitch_random_square(random.choice(list_of_grass_images))
                presence = 1
                position = align_coords(coords)
                enc_letter = np.zeros(36)
                enc_letter[char_to_int[letter]] = 1
            else:
                img_file = random.choice(list_of_grass_images)
                X = np.array(Image.open(img_file))
                presence = 0
                position = np.full(8, np.nan)
                enc_letter = np.full(36, np.nan)
        else:
            img_file = random.choice(list_of_negative_images)
            X = np.array(Image.open(img_file))
            presence = 0
            position = np.full(8, np.nan)
            enc_letter = np.full(36, np.nan)
        # preprocess input for imagenet style
        yield X, (presence, position, enc_letter)

def bg_parallel():
    def _bg_gen(gen, queue):
        g = gen()
        while True:
            queue.put(next(g))


    pqueue = multiprocessing.Queue(maxsize=100)

    p_list = [multiprocessing.Process(target=_bg_gen, args=(train_generator, pqueue)) for x in range(8)]

    [p.start() for p in p_list]
    for i in range(10000):
        yield pqueue.get()
    

def retrieve_tf_dataset():
    tf_data = tf.data.Dataset.from_generator(bg_parallel, output_types = (tf.float32,(tf.float32,tf.float32,tf.float32)), output_shapes = ((1000,1000,3),((),(8),(36))))

    tf_data = tf_data.prefetch(buffer_size = 200)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.resize(image, (224, 224)), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_contrast(image, 0.8, 1.2), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_brightness(image, 40,), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_saturation(image, 0.8, 1.2), Y)))
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_hue(image, 0.05), Y)))

    tf_data = tf_data.batch(32)
    return tf_data