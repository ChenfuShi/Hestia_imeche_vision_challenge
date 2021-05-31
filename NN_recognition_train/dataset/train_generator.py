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
SPIKE_IN_PROB = 0.01

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


list_of_grass_images = glob.glob(DATASET_DIR + "/*jpeg")
list_of_negative_images = glob.glob(TRUE_NEGATIVES_DIR + "/*jpeg")

def align_coords(coords):
    X = np.array((coords["A_X"], coords["B_X"], coords["C_X"], coords["D_X"]))
    Y = np.array((coords["A_Y"], coords["B_Y"], coords["C_Y"], coords["D_Y"]))
    average_X = np.mean(X)
    average_Y = np.mean(Y)
    total_h = np.max(Y) - np.min(Y)
    total_w = np.max(X) - np.min(X)
    
    return np.array((average_X/1000, average_Y/1000, total_h/1000, total_w/1000))


csv_file = "../data/custom_data_1.csv"
custom_labels = pd.read_csv(csv_file, index_col = "image")
list_of_extra_images = glob.glob("../data/ADDITIONAL" + "/*jpeg")
def retrieve_extra():
    img_file = random.choice(list_of_extra_images)
    img_name = os.path.basename(img_file)
    X = np.array(Image.open(img_file))
    presence = 1
    mid_X = (custom_labels.loc[img_name,"point_a"] + custom_labels.loc[img_name,"point_c"]) / 2 
    mid_Y = (custom_labels.loc[img_name,"point_b"] + custom_labels.loc[img_name,"point_d"]) / 2 
    total_h = custom_labels.loc[img_name,"point_d"] - custom_labels.loc[img_name,"point_b"]
    total_w = custom_labels.loc[img_name,"point_c"] - custom_labels.loc[img_name,"point_a"]
    position = np.array((mid_X/1000, mid_Y/1000, total_h/1000, total_w/1000))
    enc_letter = np.zeros(36)
    enc_letter[char_to_int[custom_labels.loc[img_name,"letter"]]] = 1
    return X, (presence, position, enc_letter)



def train_generator():
    while True:
        if random.random() > SPIKE_IN_PROB:
            yield retrieve_extra()
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
                position = np.full(4, np.nan)
                enc_letter = np.full(36, np.nan)
        else:
            img_file = random.choice(list_of_negative_images)
            X = np.array(Image.open(img_file))
            presence = 0
            position = np.full(4, np.nan)
            enc_letter = np.full(36, np.nan)
        # preprocess input for imagenet style
        yield X, (presence, position, enc_letter)

def bg_parallel():
    def _bg_gen(gen, queue):
        g = gen()
        while True:
            queue.put(next(g))


    pqueue = multiprocessing.Queue(maxsize=100)

    p_list = [multiprocessing.Process(target=_bg_gen, args=(train_generator, pqueue)) for x in range(7)]

    [p.start() for p in p_list]
    for i in range(10000):
        while True:
            a = pqueue.get()
            if type(a) is tuple:
                if a[0].shape == (1000,1000,3):
                    break
            print(a)
        yield a
        
    

def retrieve_tf_dataset():
    tf_data = tf.data.Dataset.from_generator(bg_parallel, output_types = (tf.float32,(tf.float32,tf.float32,tf.float32)), output_shapes = ((1000,1000,3),((),(4),(36))))

    tf_data = tf_data.map((lambda image ,Y: (tf.image.resize(image, (400, 400)), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_contrast(image, 0.8, 1.2), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_brightness(image, 40,), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_saturation(image, 0.8, 1.2), Y)), num_parallel_calls = 6)
    tf_data = tf_data.map((lambda image ,Y: (tf.image.random_hue(image, 0.05), Y)), num_parallel_calls = 6)
    tf_data = tf_data.prefetch(buffer_size = 200)
    tf_data = tf_data.batch(32)
    return tf_data