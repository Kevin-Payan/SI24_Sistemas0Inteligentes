import tensorflow as tf
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

# Directly load the model from the specified directory
model = load_model('data')  

# Define the path to the test dataset
test_path = 'C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet_test'

# Prepare the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    seed=123,
    batch_size=16,
    image_size=(200, 200),
    color_mode="grayscale"
)

# Retrieve class names
class_names = test_dataset.class_names
print("Class names:", class_names)

# Function to normalize images
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.flip_left_right(image)
    return image, label

# Apply preprocessing to the dataset
test_dataset = test_dataset.map(preprocess)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset)
print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)

# Making predictions
predictions = model.predict(test_dataset)
predicted_indices = tf.argmax(predictions, axis=1)

# Example of using class names for the first batch in the dataset and displaying images
for images, labels in test_dataset.take(1):
    labels = labels.numpy()  # Convert tensor to numpy array for indexing
    plt.figure(figsize=(10, 10))  # Set the figure size
    for i in range(len(images)):
        ax = plt.subplot(4, 4, i + 1)  # Set up the subplot; adjust the layout as needed
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')  # Display the image
        plt.title(f"Predicted: {class_names[predicted_indices[i]]}\nActual: {class_names[labels[i]]}")
        plt.axis("off")
    plt.show()  # Display the figure with all subplots