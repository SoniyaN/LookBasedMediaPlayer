import datetime
import os
import time

import cv2
import pandas as pd

harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
#-------------------------
def recognize_face(im):
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    
    #cv2.destroyAllWindows()
    return faces

