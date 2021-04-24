#!/usr/bin/python

import signal
import sys
import RPi.GPIO as GPIO
from io import BytesIO
from picamera import PiCamera
import datetime


START_BUTTON_GPIO = 18
END_BUTTON_GPIO = 23

RECORD_LED_GPIO = 26


class camera_setup():
    def __init__(self,):
        self.is_recording = False
        self.initialized = False

    def start_recording(self):
        if (self.is_recording == False) and (self.initialized == False):
            self.initialized = True
            self.camera = PiCamera()
            self.camera.exposure_compensation = -8
            self.camera.resolution = (1000, 1000)
            self.camera.framerate = 10
        if self.is_recording == False:
            self.is_recording = True
            self.camera.start_recording(f"my_test_video{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h264",
                format='h264', quality=15)
            print("started recording")
            GPIO.output(RECORD_LED_GPIO, 1)
        else:
            print("already recording")

    def stop_recording(self):
        if self.is_recording == True:
            self.camera.stop_recording()
            self.is_recording = False
            print("stopped_recording")
            GPIO.output(RECORD_LED_GPIO, 0)
        else:
            print("no recording in progress")

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)

def start_button_pressed_callback(channel):
    print("Start Button pressed!")
    camera_1.start_recording()

def end_button_pressed_callback(channel):
    print("End Button pressed!")
    camera_1.stop_recording()

if __name__ == '__main__':

    camera_1 = camera_setup()

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
