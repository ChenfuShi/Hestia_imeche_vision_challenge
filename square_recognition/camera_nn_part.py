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


camera = camera_obj()

while(True):
    # Capture frame-by-frame
    frame = camera.retrieve_frame

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.shutdown()
cv2.destroyAllWindows()