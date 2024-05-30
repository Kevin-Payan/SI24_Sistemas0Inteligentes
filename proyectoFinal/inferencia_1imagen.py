import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import pathlib
import cv2
import serial
import time

file_path = pathlib.Path(__file__).parent.absolute()

model = tf.keras.models.load_model(filepath='models/Experimento1')

dim = (224, 224)

img_paths = [
        "./test_imgs/A/A_test.jpg",
        "./test_imgs/B/B_test.jpg",
        "./test_imgs/C/C_test.jpg",
        "./test_imgs/D/D_test.jpg",
    ]

# Configura el puerto serial
ser = serial.Serial('COM10', 9600)
time.sleep(1)

for path in img_paths:
    im_file = (file_path / path).as_posix()

    image = cv2.imread(im_file, 1)

    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    processedimage = preprocess_input(resized_image)

    cv2.imshow('Prediccion original', image)
    cv2.imshow('Prediccion transformed', processedimage)
    
    # Expand dimensions to add the batch size
    processedimage = np.expand_dims(processedimage, axis=0)

    predictions = model.predict(processedimage)
    # To find the most likely class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class index: {predicted_class}")

    # To find the probability of the predicted class
    confidence = np.max(predictions, axis=1)
    print(f"Confidence in prediction: {confidence}")

    # Send the predicted class index through serial
    ser.write(f"{predicted_class}\n".encode())

    cv2.waitKey(0)


    """
    height, width, channels = image.shape
    # Print the dimensions
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Number of Channels: {channels}")
    print(f"--------------")
    """