# Flower Recognition Through CNN Keras
This repository contains code for a Convolutional Neural Network (CNN) to classify flower images into four categories: Daisy, Sunflower, Tulip, and Rose. The model is built using the Keras library and trained on a dataset of flower images.
Table of Contents
Requirements
Dataset
Code Overview
Training the Model
Evaluating the Model
Visualization
License
Requirements
The following libraries are required to run the code:

numpy
pandas
os
warnings
scikit-learn
keras
tensorflow
cv2 (OpenCV)
tqdm
PIL (Pillow)
matplotlib
seaborn
You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas scikit-learn keras tensorflow opencv-python tqdm pillow matplotlib seaborn
Dataset
The dataset used for training and validation is the "Flowers Recognition" dataset. It can be found on Kaggle and should be downloaded and placed in the input directory. The dataset should have the following structure:

css
Copy code
input
└── flowers-recognition
    └── flowers
        ├── daisy
        ├── sunflower
        ├── tulip
        └── rose
Code Overview
The main script flower_recognition.py includes the following steps:

Importing Required Libraries: Import all necessary libraries for data processing, model building, and evaluation.
Loading and Preprocessing Data: Load images from the dataset, resize them, and assign labels.
Visualizing Random Images: Visualize some random images from the dataset with their labels.
Label Encoding and One Hot Encoding: Encode the labels and convert them to one-hot vectors.
Building the CNN Model: Build the CNN model using Keras.
Compiling the Model: Compile the model with the Adam optimizer and categorical crossentropy loss.
Training the Model: Train the model on the training set and validate it on the validation set.
Evaluating the Model: Evaluate the model's performance on the test set.
Visualizing Model Performance: Plot the accuracy and loss curves for training and validation sets.
Visualizing Predictions: Visualize some correct and misclassified predictions.
Training the Model
To train the model, run the script flower_recognition.py. The training process includes loading the data, preprocessing, and training the CNN model. The model is trained for 21 epochs with a batch size of 256.
