import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import h5py
import pathlib
import cv2

file_path = pathlib.Path(__file__).parent.absolute()

model = tf.keras.models.load_model(filepath='models/Experimento2')

width = 224
height = 224
dim = (width, height)

img_paths = [
        "./test_imgs/A/Kevin_A (1).png",
        "./test_imgs/A/A_test.jpg",
        "./test_imgs/C/C_test.jpg",
    ]

for path in img_paths:
    im_file = (file_path / path).as_posix()

    image = cv2.imread(im_file, 1)
    resized_image_to_model = cv2.resize(image, dim)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    processedimage = preprocess_input(resized_image)


    cv2.imshow('Prediccion original', image)
    cv2.imshow('Prediccion transformed', processedimage)
    cv2.waitKey(0)


