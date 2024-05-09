import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
#import h5py

##Normalizar a 200x200, checa las properties, details de las imagenes

##Train Test Split ✅
##Data Augmentation!!!
##Normalizar a -1 1 o 0 1 ✅
##Construir el Modelo ✅
##Compilar el Modelo ✅
##Entrenar con grafica ✅
##Evaluar✅
# Guardar Modelo ✅
#Implementar modelo en main 

# Define the path to the dataset
dataset_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet'

# Define the path to the test images
test_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet_test'

#Prueba
test_image = 'C:/Users/kevin/Desktop/ASL_Dataset/prueba.jpg'

#Map from 0-255 to 0-1
def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

# Dataset Split (infers class labels from the subdirectory names)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=16, #32 truena en mi laptop xd
    image_size=(200, 200),
    color_mode="grayscale"
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=16,
    image_size=(200, 200),
    color_mode="grayscale"
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    seed=123,
    batch_size=16,
    image_size=(200, 200),
    color_mode="grayscale"
)

# Get the number of classes 
num_classes = len(train_dataset.class_names)

# Apply preprocessing to the dataset
train_dataset = train_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)


# Build Model
model = models.Sequential()
# 1st convolution layer                                                    
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(200,200,1) ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization()) # Que es ?
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
# Create ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath='data', monitor='val_accuracy', save_best_only=True, mode='max')

# Training
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=3, callbacks=[tensorboard_callback, checkpoint])
# tensorboard --logdir=logs/fit  (Run in Terminal)
# http://localhost:6006/?darkMode=true (Open in Browser)

# Evaluation
test_loss, test_acc = model.evaluate(test_dataset)
print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)


"""
#Save whole model
model.save("GestureRecog_ASL_model")

#Load a model
new_model = tf.keras.models.load_model("GestureRecog_ASL_model")

#Save only weights
model.save_weights("Gesture_Recog_ASL_weights")

#Load weights
model.load_weights("Gesture_Recog_ASL_weights")

#Save & Load Architecture
json_string = model.to_json()

with open("nn_model.json", "w") as f:
    f.write(json_string)

with open("nn_model.json", "r") as f:
    loaded_json_string = f.read()

new_model = tf.keras.models.model_from_json(loaded_json_string)
print(new_model.summary())
"""