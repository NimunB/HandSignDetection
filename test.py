import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import sys

# Step Six: Classify (by sending to Model)
# Step Seven: Display Result

cap = cv2.VideoCapture(0)  # Capture object - default camera (built-in webcam) should be used.
detector = HandDetector(maxHands=1)

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G"]

while True:
    success, img = cap.read()  # Capture a frame from the webcam
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Find Hands
    if hands:
        hand = hands[0]  # Note there is only one hand
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Colored image - 3 values (r,g,b)
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]  # Bounding Box

        aspectRatio = h / w

        # Overlay image on top of white image - maximizing space of square it takes
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Overlay
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # Making prediction

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Overlay
            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # Making prediction

        # Display
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)  # Display the captured frame
    cv2.waitKey(1)
