import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt

##Normalizar a 200x200, checa las properties, details de las imagenes

##Train Test Split y ver como acceder a los mismos 
##Data Augmentation!!!
##Construir el Modelo ✅
##Compilar el Modelo ✅
##Entrenar con grafica ✅
##Evaluar✅
#Debug
#Guardar
#Implementar en main 

# Define the path to the dataset
dataset_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet'

# Define the path to the test images
test_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet_test'

#Prueba
test_image = 'C:/Users/kevin/Desktop/ASL_Dataset/prueba.jpg'


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

# Get the number of classes 
num_classes = len(train_dataset.class_names)


# Build Model
model = models.Sequential()
# 1st convolution layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(200,200) ))
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

# Training
history = model.fit(train_dataset, validation_dataset, epochs=5)

# Plotting training history 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

"""
# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)
"""
