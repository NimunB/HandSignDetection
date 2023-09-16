# Computer Vision Project: Detection and Classification of ASL hand signs

This is a project that I completed in order to introduce myself to computer vision. I used OpenCV and CVZone's Hand Tracking Module to detect and capture various hand signs from American Sign Language (ASL). I then used Google's Teachable Machine to generate a machine learning model, which I applied to classify the hand signs in real-time. My goal is to expand this project and use PyTorch to create, train, and utilize a model using transfer learning. I also plan to evaluate my model in greater depth. 

This model is trained to classify letters A to G of American Sign Language. 

### Dependencies:
- cvzone 1.6.1
- mediapipe 0.9.0.1
- Tensorflow 2.9.1

### Relevant Files and Folders:
`dataCollection.py`: This is a program which you can run to save hand signs in the designated folder. Pressing s will take a snapshot of the hand.
`test.py`: This is a program which you can run to classify your hand sign in real-time using the machine learning model. It will generate a pink rectangle around your hand and will have the letter the sign stands for above. 
`Data/`: This folder contains folders of letters A through G with ~300 images of me making that sign, which was used for training purposes.
`Model/keras_model.h5`: This file contains the machine learning model from Teachable Machine. It is trained to classify letters A to G of American Sign Language. 

## Steps:
### 1) Get Webcam set up
### 2) Detect Hand
### 3) Crop Hand and Overlay onto Square
### 4) Save Images and Data Collection
### 5) Step Five: Training Model
