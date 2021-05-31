import cv2
import signal
import sys
import RPi.GPIO as GPIO
from io import BytesIO
from picamera import PiCamera
import datetime
import time
import numpy as np

class camera_obj:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self):
        # Initialize the PiCamera and the camera image stream
        self.camera = PiCamera()
        self.camera.exposure_compensation = -8
        self.camera.resolution = (1000, 1000)
        self.camera.framerate = 10
        time.sleep(1)

    def retrieve_frame(self):
        output = np.empty((1000, 1000, 3), dtype=np.uint8)
        self.camera.capture(output, 'rgb')
        return output

    def shutdown(self):
        self.camera.close()