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


    class_indices = {
        '1' : 'B',
        '2' : 'C',
        '5' : 'F',
        '6' : 'H',
        '11': ' ',
        '13': 'V',
    }

    """
    # Verifica si la clase predicha está en la lista de valores válidos
    if predicted_class in ['1', '2', '5', '6', '7', '8']:
        ser.write(f"{predicted_class}\n".encode())  # Añade un salto de línea para la claridad en la transmisión
        print(f"Enviado: {predicted_class}")
    else:
        print("Carácter no válido.")
    """

    # Check if the predicted class index exists in the dictionary
    if predicted_class in class_indices:
        # Retrieve the corresponding class label from the dictionary
        class_label = class_indices[predicted_class]
        # Send the class label through the serial
        ser.write(class_label.encode())  # Add a newline for clarity in transmission
        print(f"Enviado: {class_label}")
    else:
        print("Carácter no válido.")