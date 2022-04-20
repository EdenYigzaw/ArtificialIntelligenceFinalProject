#Face Detection inspired by https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras

#Declare cascade classifier used for Face Detection (Uses the pre-trained Haar Cascade Classifier)

frontFaceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Declare Model to be used for Emotion Recognition and Emotion Classes

maskModel = keras.models.load_model("fer_model_M1")
class_names = ['WithMask', 'WithoutMask']

#Get default webcam source
webcamSource = cv2.VideoCapture(0)
webcamSource.set(3,500) #Set display window Width
webcamSource.set(4,500) #Set display window Height

#Main while loop variable
capturingVideo = True

while capturingVideo:
    
    #Read one frame of the video
    _, videoFrame = webcamSource.read()

    #Search for face and draw a rectangle on its location
    gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.15
    minNeighbors = 6
    faces = frontFaceClassifier.detectMultiScale(gray, scaleFactor, minNeighbors)

    for(x, y, width, height) in faces:
        #Draw a rectangle
        topLeftRectangle = (x,y) #Top left coordinates of expected rectangle
        recWidth = (x + width)
        recHeight = (y + height)
        bottomRightRectangle = (recWidth, recHeight) #Bottom right coordinates of expected rectangle
        color = (255,0,0)
        colorThickness = 3

        cv2.rectangle(videoFrame, topLeftRectangle, bottomRightRectangle, color, colorThickness)
        
        grayFrame = cv2.resize(gray[y:recHeight, x:recWidth], (48,48))
        imgReformat = np.expand_dims(np.array(grayFrame), axis = 0)

        maskDecision = maskModel.predict(imgReformat)
        
        classIndex = np.where(maskDecision[0] == np.amax(maskDecision[0]))
        print(class_names[classIndex[0][0]])




    #Create and display a window showing the webcam video
    cv2.imshow('Display Window', videoFrame)
    cv2.waitKey(1)


    #If user clicks on the x button in the display frame, stop video capture
    if cv2.getWindowProperty('Display Window', cv2.WND_PROP_VISIBLE) < 1 :
        capturingVideo = False


#Release capture and close any associated windows
webcamSource.release()
cv2.destroyAllWindows()