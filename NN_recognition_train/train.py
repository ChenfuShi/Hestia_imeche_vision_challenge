import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as k
import glob
from PIL import Image
import argparse
import sys

from dataset.train_generator import retrieve_tf_dataset
from model.mobile_net import retrieve_mobilenet_model

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)


if __name__ == '__main__':

    model_name = sys.argv[1]

    tf_data = retrieve_tf_dataset()

    model = retrieve_mobilenet_model()

    model.fit(tf_data, epochs = 1, verbose = 2,)

    model.trainable = True

    model.fit(tf_data, epochs = 10, verbose = 2,)
    
    model.save(f'../weights/{model_name}.tf', save_format = "tf")
