Gender Detection using OpenCV and Deep Learning

Overview

This project implements real-time gender detection using OpenCV's Deep Neural Network (DNN) module. It utilizes pre-trained Caffe models to detect faces and classify gender as Male or Female.

Features

Uses OpenCV DNN for face detection.

Classifies gender as Male or Female.

Works with pre-trained Caffe models.

Optimized for Google Colab (uses cv2_imshow() instead of cv.imshow()).

Dependencies

Ensure you have the following libraries installed:

pip install opencv-python numpy gdown

Model Download

The required pre-trained models are downloaded automatically using:

!gdown https://drive.google.com/uc?id=1_aDScOvBeBLCn_iv0oxSO8X1ySQpSbIS
!unzip modelNweight.zip

File Structure

├── modelNweight/
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
├── gender_detection.py (Main Script)
├── README.md (This file)

Usage

Run the script in Google Colab:

!gdown https://drive.google.com/uc?id=1_aDScOvBeBLCn_iv0oxSO8X1ySQpSbIS
!unzip modelNweight.zip

Import required libraries:

import cv2 as cv
import numpy as np
import time
from google.colab.patches import cv2_imshow

Run the gender detection model:

input_img = cv.imread("Photo-1.jpeg")
if input_img is None:
    print("Error: Image not found or could not be loaded.")
else:
    output_img = gender_detector(input_img)
    cv2_imshow(output_img)  # Display image in Colab

How It Works

Face Detection: The OpenCV DNN model detects faces in the input image.

Gender Classification: The detected face is passed through a gender classification model.

Output Display: The image is displayed with the predicted gender label.

Sample Output

An image with the detected face and gender label drawn on it.

Notes

Works best with frontal face images.

The model may have some biases due to the dataset it was trained on.

License

This project is for educational purposes. The models used are publicly available.

Developed by Sayan 🚀
