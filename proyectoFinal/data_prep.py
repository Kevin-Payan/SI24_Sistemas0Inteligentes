import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import pathlib
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Mandar a GPU Memory
device = tf.config.experimental.list_physical_devices('GPU')
print('GPUS: ', len(device))
tf.config.experimental.set_memory_growth(device[0], True)

file_path = pathlib.Path(__file__).parent.absolute()

print(file_path)
train_path = 'dataset/train'
train_path = (file_path / train_path).as_posix()
val_path = 'dataset/val'
val_path = (file_path / val_path).as_posix()
test_path = 'test_imgs'
test_path = (file_path / test_path).as_posix()
print(test_path)

def join_batches(original_batch, augmented_batch):
    while True:
        original_images, original_labels = next(original_batch)
        augmented_images, augmented_labels = next(augmented_batch)
        combined_images = np.concatenate((original_images, augmented_images)) # Juntar imagenes
        combined_labels = np.concatenate((original_labels, augmented_labels)) # Juntar labels
        yield combined_images, combined_labels

#Configuramos los parametros para el data augmentation
augment_params= tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    horizontal_flip=True,  # Flip horizontal
    rotation_range=20  # Rotacion Random
)    


#Hacemos preprocessing de las imagenes y los ponemos en batches
#Lo separamos en dos para dejar las imagenes que ya teniamos y solamente aumentarle 
#mas pero con data augmentation
train_batches_original = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224, 224), 
                         classes=['A', 'B', 'C', 'D', 'del', 'F', 'H', 'I', 'L',
                        'nothing', 'P', 'space', 'U', 'V', 'W', 'Y'], batch_size=10)

train_batches_augmented = augment_params.flow_from_directory(
                        directory=train_path, target_size=(224, 224), 
                         classes=['A', 'B', 'C', 'D', 'del', 'F', 'H', 'I', 'L',
                        'nothing', 'P', 'space', 'U', 'V', 'W', 'Y'], batch_size=10)

#train_batches = join_batches(train_batches_original, train_batches_augmented)

val_batches = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=val_path, target_size=(224, 224), 
                         classes=['A', 'B', 'C', 'D', 'del', 'F', 'H', 'I', 'L',
                        'nothing', 'P', 'space', 'U', 'V', 'W', 'Y'], batch_size=10)  
test_batches = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224, 224), 
                         classes=['A', 'B', 'C', 'D', 'del', 'F', 'H', 'I', 'L',
                        'nothing', 'P', 'space', 'U', 'V', 'W', 'Y'], batch_size=10, shuffle=False)  



imgs, labels = next(train_batches_original) #Como es batch de 10, 10 imagenes seran

def plotImages(images_array):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_array,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)

