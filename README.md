# COVID 19 Face Mask Detector [![Made with Python](https://img.shields.io/badge/python-3.5.2-grey?style=for-the-badge&labelColor=yellow&logo=python)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.1-grey?style=for-the-badge&labelColor=blue&logo=tensorflow)](https://www.tensorflow.org/)<br>

### What is this project?
This project works on the concepts of machine learning and is able to detect whether a person, in front of the camera (Real-Time VideoFeed), is wearing a mask or not. The primary objective to create this project is to make and Edge AI device (Nvidia Jetson Nano) that can be placed in crowded areas or shops to warn the owners of any possible threats in the "NEW NORMAL".

## Pre-Requisites Before running
* Install Tensorflow using `pip install tensorflow`.
* Install OpenCV using `pip install opencv-python` and `pip install opencv-contrib-python`.
* Install Keras using `pip install keras`.
* Install Numpy using `pip install numpy`.


## About the Dataset
### Dataset Creation
The major problem that was faced during the creation of this machine learning model was to get adequate data for the model to be trained upon. After scrounging the internet for hours, I found this article by Dr. Adrian Rosebrock which explained a magic trick to create this dataset. Why a magic trick? Find out yourself ! Go ahead and read this article and show some well-deserved love to his blog. Without his article, this project would not have been possible !
<br><br>
[Dr. Rosebrock's Article : COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
### Dataset Contents
The dataset consists of 1376 images belonging to the following two classes:
* with_mask: 690 images
* without_mask: 686 images
### Training the Model
The model can be trained using the train_mask_detector.py script using the following command

    python train_mask_detector.py --dataset dataset
    
### Accuracy and Loss Curves
<p align="center">
<img src="plot.png" width="450" height="350">
</p>

## Inference
Inference can be performed by using the inference.py script using the following command.
    
    python inference.py
    
## Results
See for yourself !
<p align="center">
<img src="result.gif" width="450" height="350">
</p>
