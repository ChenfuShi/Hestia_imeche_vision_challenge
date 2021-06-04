import cv2
import signal
import sys
import RPi.GPIO as GPIO
from io import BytesIO
from picamera import PiCamera
import datetime
import time
import numpy as np
import threading
import pandas as pd

from camera_class import camera_obj

class NN_recognition():
    def __init__(self):
        self.state = "off"

    def initialize_script(self):   
        self.state = "on"     
        self.camera = camera_obj()

        # dronekit stuff


        # results
        self.results = []

        thread = threading.Thread(target=self.run_loop, args=())
        thread.daemon = True                            # Daemonize thread (allow main to die)
        thread.start()                                  # Start the execution

    def end_script(self):
        self.state = "off"
        time.sleep(0.5) # wait for run_loop to terminate

        self.camera.shutdown()
        # run finalization and send results back

    def run_loop(self):
        while(True):
            # Capture frame-by-frame
            frame = self.camera.retrieve_frame()

            # retrieve dronekit telemetry

            # run neural network 1


            # if result from neural network 1 is good then run neural network 2



            # save results

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or self.state == "off":
                self.state = "off"
                break
        
        cv2.destroyAllWindows()
        

    def finalize_results(self):
        pass



