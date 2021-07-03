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
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)
from model.step_2_GAP import train_this




if __name__ == '__main__':

    model_name = sys.argv[1]

    print(f"MODEL NAME = {model_name}")

    train_this(model_name)
