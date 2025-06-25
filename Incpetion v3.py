import os
import cv2
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, concatenate, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from kerastuner import BayesianOptimization
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load metadata
metadata_path = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\HAM10000_metadata.csv'  
metadata_df = pd.read_csv(metadata_path)

data_dir = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\reorganized'
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Defining the Image size
Image_Size = 299  # Adjusted for Inception-v3
# Dividing this in individual classes with 500 sample data to avoid overfitting
num_samples_per_class = 500
images_data = []
labels = []
metadata = []

for c in classes:
    folder = os.path.join(data_dir, c)
    label = classes.index(c)
    images = os.listdir(folder)
    random.shuffle(images)  # Shuffle the list of images
    selected_images = images[:num_samples_per_class]  # Select the first 500 images
    print(f"Class: {c}, Number of sampled images: {len(selected_images)}")
    for lesion_id in selected_images:
        image_path = os.path.join(folder, lesion_id)
        image = cv2.imread(image_path)  # Read the image
        image = cv2.resize(image, (Image_Size, Image_Size))  # Resize the image to fit Inception-v3
        images_data.append(image)  # Append image to the images list
        labels.append(label)  # Append label to the labels list
        # Extract metadata for the current image
        image_metadata = metadata_df.loc[metadata_df['lesion_id'] == lesion_id.split('.')[0]]
        metadata.append(image_metadata)  # Append metadata to the metadata list

# Convert lists to numpy arrays
images_data = np.array(images_data)
labels = np.array(labels)
metadata = np.array(metadata)

# Shuffle the data
shuffle_indices = np.arange(len(images_data))
np.random.shuffle(shuffle_indices)
images_data = images_data[shuffle_indices]
labels = labels[shuffle_indices]
metadata = metadata[shuffle_indices]

# Separate features (X) and labels (y)
X_images = images_data
X_metadata = metadata
y = labels

# Scaling down the pixel
X_images = X_images / 255

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

# Define sensitivity and specificity functions
def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(y_true, axis=-1), 1), tf.math.equal(tf.argmax(y_pred, axis=-1), 1)), dtype=tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), 1), dtype=tf.float32))
    return true_positives / (actual_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(y_true, axis=-1), 0), tf.math.equal(tf.argmax(y_pred, axis=-1), 0)), dtype=tf.float32))
    actual_negatives = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), 0), dtype=tf.float32))
    return true_negatives / (actual_negatives + tf.keras.backend.epsilon())

# Load InceptionV3 model with pretrained weights
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(Image_Size, Image_Size, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier on top
x = base_model.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(classes), activation='softmax')(x)

# Create the model
model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', sensitivity, specificity])

# Define the image data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the Bayesian optimization tuner
tuner = BayesianOptimization(
    model,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='my_dir',
    project_name='HAM_BayesianOpt'
)

# Perform the hyperparameter search with augmentation
tuner.search(datagen.flow(X_train, y_train), epochs=10, validation_data=(X_test, y_test))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the model with augmented data
history = best_model.fit(datagen.flow(X_train, y_train), epochs=20, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_acc, test_sensitivity, test_specificity = best_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
print("Test Sensitivity:", test_sensitivity)
print("Test Specificity:", test_specificity)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
