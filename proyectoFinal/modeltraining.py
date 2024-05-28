import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from data_prep import train_batches, val_batches, test_batches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import h5py

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##Normalizar a 200x200, checa las properties, details de las imagenes

##Train Test Split ✅
##Data Augmentation!!! (Primero prueben jugar con la iluminacion porque el dataset es mas oscuro)
##Normalizar a -1 1 o 0 1 ✅
##Construir el Modelo ✅
##Compilar el Modelo ✅
##Entrenar con grafica (Tensorboard) ✅
##Evaluar✅
# Guardar Modelo ✅
#Implementar modelo en main 


# Build Model

base_model = tf.keras.applications.resnet50.ResNet50(
    weights='imagenet', #Por default esta el imagenet
    input_shape=(224, 244, 3),
    include_top=False #Le quitamos la ultima capa
) 

base_model.trainable = False #Congelamos los pesos


vgg16_model = tf.keras.applications.vgg16.VGG16() 

vgg16_model.summary()
print(type(vgg16_model))


new_model = tf.keras.models.Sequential()
for layer in vgg16_model.layers[:-1]:
    new_model.add(layer) # Anadimos todas las layers menos la ultima
    #print('Se añadio la layer!', layer)

for layer in new_model.layers:
    layer.trainable = False #Congelar los pesos del new model

new_model.add(Dense(units=16, activation='softmax')) #Le añadimos la ultima capa para 16 predicciones
                                                    # Es una fully connected

# Nos preparamos para entrenamiento, definimos optimizer y funcion de costo
""" optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
#Entrenamos
new_model.fit(x=train_batches, validation_data = val_batches, epochs = 5, verbose = 1)

new_model.save('models/Experimento1') """

#Predicciones
#Primero, cargamos el modelo
loaded_model = tf.keras.models.load_model(filepath='models/Experimento1')
preds = loaded_model.predict(x=test_batches, verbose=1)
clases = test_batches.classes #Agarramos las clases 
cm = confusion_matrix(y_true=clases, y_pred=np.argmax(preds, axis=-1))
print(test_batches.class_indices)
cm_plot_labels = ['A', 'B', 'C', 'D', 'del', 'F', 'H', 'I', 'L', 'nothing',
                  'P', 'space', 'U', 'V', 'W', 'Y']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# Compile Model
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Setup TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 

# Create ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath='data', monitor='val_accuracy', save_best_only=True, mode='max')

# Training
#history = model.fit(train_dataset, validation_data=validation_dataset, epochs=15, callbacks=[tensorboard_callback, checkpoint])
# tensorboard --logdir=logs/fit  (Run in Terminal)
# http://localhost:6006/?darkMode=true (Open in Browser)

# Evaluation
#test_loss, test_acc = model.evaluate(test_dataset)
#print('\nTest loss:', test_loss)
#print('Test accuracy:', test_acc)


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