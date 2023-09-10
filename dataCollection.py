import cv2
from cvzone.HandTrackingModule import HandDetector

# Step One: Webcam
# Step Two: Detect Hand
# Step Three: Crop Hand

cap = cv2.VideoCapture(0)  # Capture object - default camera (built-in webcam) should be used.
detector = HandDetector(maxHands=1)
while True:
    success, img = cap.read()  # Capture a frame from the webcam
    hands, img = detector.findHands(img) # Find Hands
    cv2.imshow("Image", img)  # Display the captured frame
    cv2.waitKey(1)  # 1 second delay
