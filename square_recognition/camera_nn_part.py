from typing_extensions import ParamSpecArgs
import cv2
import signal
import sys
import RPi.GPIO as GPIO
from io import BytesIO
from picamera import PiCamera
import datetime
import time
import numpy as np

from camera_class import camera_obj

class NN_recognition():
    def __init__(self):
        self.state = "off"

    def initialize_script(self):   
        self.state = "on"     
        camera = camera_obj()
        while(True):
            # Capture frame-by-frame
            frame = camera.retrieve_frame()

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or self.state == "off":
                break
        camera.shutdown()
        cv2.destroyAllWindows()

    def end_script(self):
        self.state = "off"





