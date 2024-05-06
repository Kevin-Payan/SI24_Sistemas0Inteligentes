import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime


##Normalizar a 200x200, checa las properties, details de las imagenes

##Train Test Split ✅
##Data Augmentation!!!
##Normalizar a -1 1 o 0 1 
##Construir el Modelo ✅
##Compilar el Modelo ✅
##Entrenar con grafica ✅
##Evaluar✅
#Guardar
#Implementar en main 

# Define the path to the dataset
dataset_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet'

# Define the path to the test images
test_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet_test'

#Prueba
test_image = 'C:/Users/kevin/Desktop/ASL_Dataset/prueba.jpg'

"""
#Prueba
test_image = 'C:/Users/kevin/Desktop/ASL_Dataset/prueba.jpg'

# Load an image
img = mpimg.imread(test_image)
# Get image dimensions
height, width, channels = img.shape

print(f'Width: {width}, Height: {height}, Channels: {channels}')
"""

# Training & Validation Split (infers class labels from the subdirectory names)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=16 #32 truena en mi laptop xd
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=16 
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    seed=123,
    batch_size=3 
)

# Get the number of classes 
num_classes = len(train_dataset.class_names)


# Build Model
model = models.Sequential()
# 1st convolution layer                                                     #200x200 (tamano) x RGB DUDA TAMANO 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(200,200,3) ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization()) #Que es batch normalization?
# 2nd convolution layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# 3rd convolution layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# fully-connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))


# Print model summary
model.summary()

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Setup TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5, callbacks=[tensorboard_callback])
# tensorboard --logdir=logs/fit  (correr en terminal)


# Evaluation
test_loss, test_acc = model.evaluate(test_dataset)
print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)

