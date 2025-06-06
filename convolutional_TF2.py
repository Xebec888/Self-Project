"""Classify color images (32x32 pixels, 3 channels) into 10 categories using a CNN for any image dataset."""
import numpy as np
import tensorflow as tf
import pickle
import random
from tensorflow.keras import layers, models

# Set parameters
image_size = 32   #image_size matches your new dataset's height and width.
num_channels = 3  #num_channels (e.g., 1 for grayscale, 3 for RGB)
num_categories = 10 #number of distinct labels in the dataset
num_filters = 32
filter_size = 5
num_epochs = 200
batch_size = 10

# Load training data
train_data = []
train_labels = []

for file_index in range(5):
    with open(f'location of file{file_index+1}', 'rb') as train_file:
        train_dict = pickle.load(train_file, encoding='latin1')
        train_data.append(train_dict['data'])
        train_labels.extend(train_dict['labels'])

#preprocess training data and labels
train_data = np.concatenate(train_data).astype(np.float32) / 255.0
train_data = train_data.reshape([-1, num_channels, image_size, image_size])
train_data = train_data.transpose([0, 2, 3, 1])
train_labels = tf.keras.utils.to_categorical(train_labels, num_categories) #using categorical to label

# Load test data and preprocess the test data
with open('location of file', 'rb') as test_file:
    test_dict = pickle.load(test_file, encoding='latin1')
test_data = test_dict['data'].astype(np.float32) / 255.0
test_data = test_data.reshape([-1, num_channels, image_size, image_size])
test_data = test_data.transpose([0, 2, 3, 1])
test_labels = tf.keras.utils.to_categorical(test_dict['labels'], num_categories)

# Build the model
model = models.Sequential([
    layers.Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=(image_size, image_size, num_channels)), #input_shape in the first layer sets the input and removing need for placeholders
    layers.Dropout(0.4),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(num_filters, filter_size, padding='same', activation='relu'),
    layers.Dropout(0.4),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(num_filters, filter_size, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(num_filters, filter_size, padding='same', activation='relu'),
    layers.Dropout(0.4),
    layers.Flatten(), # flatten the data after conv, drop and pooling
    layers.Dense(512, activation='relu'), # create connected layers
    layers.BatchNormalization(), #stabilize and accelerate training by normalizing the layer’s outputs before they go to the next layer.
    layers.Dense(num_categories)
])

# Compile the model (optimizer and loss)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=2) #verbose determine how much output u see in the console during training

# Evaluate the model ( success rate)
loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print(f'Accuracy: {accuracy:.4f}')

#saving the model
model.save('current_directory.h5 ')

