import numpy as np
import cv2
import time
from keras.applications.vgg16 import preprocess_input

def serial_prediction(image,model,ser):

    #time.sleep(1)

    dim = (224, 224)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    processedimage = preprocess_input(resized_image)
        
    # Expand dimensions to add the batch size
    processedimage = np.expand_dims(processedimage, axis=0)

    predictions = model.predict(processedimage)
    # Find the most likely class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class index: {predicted_class}")

    # Probability of the predicted class
    confidence = np.max(predictions, axis=1)
    print(f"Confidence in prediction: {confidence}")

    predicted_class = str(predicted_class)

    # Send the predicted class index through serial
    # ser.write(f"{predicted_class}\n".encode())

    valid_numbers = ["1" , "2" , "5" , "6" , "11" , "13"]

    if predicted_class in valid_numbers :
        ser.write(f"{predicted_class}\n".encode())
        time.sleep(2)
        
    



  