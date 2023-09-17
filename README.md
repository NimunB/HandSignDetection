# Computer Vision Project: Detection and Classification of ASL hand signs

This is a project that I completed in order to introduce myself to computer vision. I used OpenCV and CVZone's Hand Tracking Module to detect and capture various hand signs from American Sign Language (ASL). I then used Google's Teachable Machine to generate a machine learning model, which I applied to classify the hand signs in real-time. My goal is to expand this project and use PyTorch to create, train, and utilize a model using transfer learning. I also plan to evaluate my model in greater depth. 

This model is trained to classify letters A to G of American Sign Language. 

### Dependencies:
- cvzone 1.6.1
- mediapipe 0.9.0.1
- Tensorflow 2.9.1

### Relevant Files and Folders:
- `dataCollection.py`: This is a program which you can run to save hand signs in the designated folder. Pressing s will take a snapshot of the hand.
- `test.py`: This is a program which you can run to classify your hand sign in real-time using the machine learning model. It will generate a pink rectangle around your hand and will have the letter the sign stands for above. 
- `Data/`: This folder contains folders of letters A through G with ~300 images of me making that sign, which was used for training purposes.
- `Model/keras_model.h5`: This file contains the machine learning model from Teachable Machine. It is trained to classify letters A to G of American Sign Language. 

## Steps:
### 1) Detect Hand
<img width="1440" alt="Screen Shot 2023-09-10 at 12 45 55 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/94bd7db9-c552-4653-a0bd-1a4faf002e98">
<p align="center"><i>Using CVZone's Hand Tracking Module</i></p>

### 2) Crop Hand and Overlay onto Square
<img width="1438" alt="Screen Shot 2023-09-10 at 2 57 35 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/60b5c83f-bec8-4496-b250-2ed68033d7eb">
<p align="center"><i>Cropping to square</i></p>

<img width="1119" alt="Screen Shot 2023-09-10 at 11 14 45 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/6c5f5a7f-f2a6-4828-8e9f-38988860c90f">
<p align="center"><i>Filling background with white to stabilize dataset</i></p>

### 3) Save Images and Data Collection 
Ran `dataCollection.py` and clicked s to capture around 300 images of each hand sign, from A through G. 

### 4) Training Model

<img width="1284" alt="Screen Shot 2023-09-10 at 11 52 48 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/3f17c9e1-5c4d-4e8c-8984-24a84a026e34">

### 5) Apply Model

<img width="1440" alt="Screen Shot 2023-09-16 at 10 33 14 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/25a3c8e1-ee75-4e27-ac42-6c2aba15e69e">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 33 23 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/50c8f5f2-4fa0-41d8-8a52-0a7c9cd468b3">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 33 30 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/2fab3ef6-3ef6-43ee-9e49-053827b4cbbc">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 33 39 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/50af9ff0-8120-4d19-aa6a-ad0afee28d98">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 33 54 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/bff42518-d4df-4ff7-8738-12f0bc455eab">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 34 45 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/d86d5ae8-4db4-4adc-a08b-33e1ba1af2b8">
<img width="1440" alt="Screen Shot 2023-09-16 at 10 36 43 PM" src="https://github.com/NimunB/HandSignDetection/assets/32827637/4c184e4a-e835-4774-b43d-ffec822cc70c">

Overall, the model achieved a 79% accuracy for this classification problem.
