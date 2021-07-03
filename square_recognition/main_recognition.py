import cv2
import signal
import sys
import RPi.GPIO as GPIO
from io import BytesIO
import datetime
import time
import numpy as np

from camera_nn_part import NN_recognition


###########
# steps
# camera module
# predict position
# identify colour
# retrieve coordinates
# use some clustering algorithm and identify best 4
# send results
# communication


START_BUTTON_GPIO = 18
END_BUTTON_GPIO = 23

RECORD_LED_GPIO = 26


def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)

def start_button_pressed_callback(channel):
    print("Start Button pressed!")
    if NN_class.state == "off":
        GPIO.output(RECORD_LED_GPIO, 1)
        NN_class.initialize_script()

def end_button_pressed_callback(channel):
    print("End Button pressed!")
    if NN_class.state == "on":
        NN_class.end_script()
        GPIO.output(RECORD_LED_GPIO, 0)


if __name__ == '__main__':

    NN_class = NN_recognition()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(START_BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(END_BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.setup(RECORD_LED_GPIO, GPIO.OUT)
    GPIO.output(RECORD_LED_GPIO, 0)

    GPIO.add_event_detect(START_BUTTON_GPIO, GPIO.RISING, 
            callback=start_button_pressed_callback, bouncetime=100)
    GPIO.add_event_detect(END_BUTTON_GPIO, GPIO.RISING, 
            callback=end_button_pressed_callback, bouncetime=100)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
