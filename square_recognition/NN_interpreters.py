import os
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


compiled_NN_path = ""


class NN_coral():
    def __init__(self):
        self.interpreter = Interpreter(model_path=compiled_NN_path,
                                        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        self.interpreter.allocate_tensors()