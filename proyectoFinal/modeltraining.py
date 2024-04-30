import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path to the dataset
dataset_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet'

# Training Split
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=16 #32 truena en mi laptop xd
)

# Validation Split
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=16 
)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(5):  # Display 5 images
        ax = plt.subplot(3, 3, n + 1)
        plt.imshow(image_batch[n].numpy().astype("uint8"))
        plt.title(label_batch[n])
        plt.axis("off")
    plt.show()

# Extract a batch of images and labels from the training dataset
for image_batch, label_batch in train_dataset.take(1):
    show_batch(image_batch, label_batch)