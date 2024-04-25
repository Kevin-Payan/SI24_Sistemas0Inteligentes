import tensorflow as tf

# Create a Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer; assumes input data is 28x28
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    tf.keras.layers.Dropout(0.2),                   # Dropout layer for regularization
    tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 classes (for classification)
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Assume x_train and y_train are your data and labels respectively
model.fit(x_train, y_train, epochs=5)



# Assume x_test and y_test are your test data and labels
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)



predictions = model.predict(x_test)
