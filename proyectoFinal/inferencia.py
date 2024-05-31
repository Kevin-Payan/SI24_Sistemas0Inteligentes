import numpy as np
import cv2
import time
from keras.applications.vgg16 import preprocess_input

def serial_prediction(image, model, ser):

    #time.sleep(1)

    # Dimensiones deseadas para la entrada del modelo
    dim = (224, 224)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    processed_image = preprocess_input(resized_image)
        
    # Expande las dimensiones para agregar el tamaño del batch
    processed_image = np.expand_dims(processed_image, axis=0)

    # Realiza la predicción con el modelo
    predictions = model.predict(processed_image)
    
    # Encuentra la clase más probable
    predicted_class_array = np.argmax(predictions, axis=1)
    predicted_class = str(predicted_class_array[0])  # Extrae el primer elemento y conviértelo a cadena
    print(f"Predicted class index: {predicted_class}")

    # Probabilidad de la clase predicha
    confidence = np.max(predictions, axis=1)
    print(f"Confidence in prediction: {confidence}")

    # Verifica si la clase predicha está en la lista de valores válidos
    if predicted_class in ['1', '2', '5', '11', '13', '6']:
        ser.write(f"{predicted_class}\n".encode())  # Añade un salto de línea para la claridad en la transmisión
        print(f"Enviado: {predicted_class}")
    else:
        print("Carácter no válido.")
