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
imgSize = 30

counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G"]

while True:
    success, img = cap.read()  # Capture a frame from the webcam
    hands, img = detector.findHands(img)  # Find Hands
    if hands:
        hand = hands[0]  # Note there is only one hand
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Colored image - 3 values (r,g,b)
        imgCrop = img[y - offset:y + h + offset, x - offset: x + w + offset]  # Bounding Box

        aspectRatio = h / w

        # Overlay image on top of white image - maximizing space of square it takes
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Overlay
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)
            sys.stdout.flush()
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Overlay

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)  # Display the captured frame
    cv2.waitKey(1)
