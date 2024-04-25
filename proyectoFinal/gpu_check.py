#Acuerdate de activar el entorno de ml_env antes de abrir este proyecto

import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# List all physical devices visible to TensorFlow
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# Specifically check for a GPU
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

# If a GPU is available, print its details
if gpus:
    for gpu in gpus:
        print("GPU:", gpu.name)
        # Fetch and print GPU details
        gpu_details = tf.config.experimental.get_device_details(gpu)
        print("GPU details:", gpu_details)
else:
    print("No GPU found. TensorFlow is using CPU.")



