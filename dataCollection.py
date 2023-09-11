import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Step One: Webcam
# Step Two: Detect Hand
# Step Three: Crop Hand and Overlay onto Square
# Step Four: Save Images and Data Collection

cap = cv2.VideoCapture(0)  # Capture object - default camera (built-in webcam) should be used.
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 30

folder = "Data/G"
counter = 0

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
    key = cv2.waitKey(1) & 0xFF  # Mask to isolate the key code
    # Save on s click
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # f string
        print(counter)