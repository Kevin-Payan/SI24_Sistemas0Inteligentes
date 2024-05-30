import numpy as np
import cv2
import serial
import time
from keras.applications.vgg16 import preprocess_input

def serial_prediction(image,model):

    # Configura el puerto serial
    ser = serial.Serial('COM10', 9600)

    #Si no tiene suficiente tiempo no va a servir nada
    time.sleep(1)

    dim = (224, 224)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    processedimage = preprocess_input(resized_image)

    #Borrar Imshow
    #cv2.imshow('Prediccion original', image)
    #cv2.imshow('Prediccion transformed', processedimage)
        
    # Expand dimensions to add the batch size
    processedimage = np.expand_dims(processedimage, axis=0)

    predictions = model.predict(processedimage)
    # Find the most likely class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class index: {predicted_class}")

    # Probability of the predicted class
    confidence = np.max(predictions, axis=1)
    print(f"Confidence in prediction: {confidence}")

    # Send the predicted class index through serial
    ser.write(f"{predicted_class}\n".encode())

    #cv2.waitKey(0)


  