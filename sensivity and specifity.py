import os
import cv2
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, Flatten,Dense,BatchNormalization
from sklearn.model_selection import train_test_split
from kerastuner.tuners import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load metadata
metadata_path = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\HAM10000_metadata.csv'  
metadata_df = pd.read_csv(metadata_path)

data_dir = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\reorganized'
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
#Defining the Image size
Image_Size = 224
#dividing this in individual classes with 500 sample data to avoid overfitting
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
        image = cv2.resize(image, (Image_Size, Image_Size))  # Resize the image
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
X = {'images': images_data, 'metadata': metadata}
y = labels

X_images = images_data
X_metadata = metadata
y = labels

# Check the shapes
print("X_images shape:", X_images.shape)
print("X_metadata shape:", X_metadata.shape)
print("y shape:", y.shape)

# Scaling down the pixel
X_images = X_images / 255

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

#Designing individual models for image classification
#AlexNet architecture


def create_alexnet(hp):
    num_classes = len(classes)
    model1 = Sequential()
    model1.add(BatchNormalization())
    # Layer 1
    model1.add(Conv2D(hp.Int('conv1_filters', min_value=32, max_value=256, step=32),
                      kernel_size=(11, 11),
                      strides=(4, 4),
                      activation='relu',
                      input_shape=(227, 227, 3)))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 2
    model1.add(Conv2D(hp.Int('conv2_filters', min_value=32, max_value=256, step=32),
                      kernel_size=(5, 5),
                      padding='same',
                      activation='relu'))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model1.add(BatchNormalization())
    # Layer 3
    model1.add(Conv2D(hp.Int('conv3_filters', min_value=32, max_value=256, step=32),
                      kernel_size=(3, 3),
                      padding='same',
                      activation='relu'))
    model1.add(BatchNormalization())
    # Layer 4

    model1.add(Conv2D(hp.Int('conv4_filters', min_value=32, max_value=256, step=32),
                      kernel_size=(3, 3),
                      padding='same',
                      activation='relu'))
    model1.add(BatchNormalization())
    # Layer 5
    model1.add(Conv2D(hp.Int('conv5_filters', min_value=32, max_value=256, step=32),
                      kernel_size=(3, 3),
                      padding='same',
                      activation='relu'))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model1.add(BatchNormalization())
    # Flatten
    model1.add(Flatten())
        # Layer 6
    model1.add(Dense(hp.Int('dense1_units', min_value=32, max_value=1024, step=32),
                    activation='relu'))
    model1.add(Dropout(0.5))
    # Layer 7
    model1.add(Dense(hp.Int('dense2_units', min_value=32, max_value=1024, step=32),
                    activation='relu'))
    model1.add(Dropout(0.5))
        # Output layer
    model1.add(Dense(num_classes, activation='softmax'))
    
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', sensitivity, specificity])

    return model1

# Define sensitivity and specificity functions
def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(y_true, axis=-1), 1), tf.math.equal(tf.argmax(y_pred, axis=-1), 1)), dtype=tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), 1), dtype=tf.float32))
    return true_positives / (actual_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(y_true, axis=-1), 0), tf.math.equal(tf.argmax(y_pred, axis=-1), 0)), dtype=tf.float32))
    actual_negatives = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), 0), dtype=tf.float32))
    return true_negatives / (actual_negatives + tf.keras.backend.epsilon())
                                                                                   

# Define the tuner
tuner = BayesianOptimization(
    create_alexnet,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='my_dir',
    project_name='ham10000_alexnet_tuning',overwrite=True)

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=1, validation_split=0.2)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Re-instantiate the model with the best hyperparameters and train it on the entire training set
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=25, validation_split=0.2)
test_loss, test_acc, test_sensitivity, test_specificity = best_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
print("Test Sensitivity:", test_sensitivity)
print("Test Specificity:", test_specificity)

# Train the model and capture the history
history = best_model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# Get training history
history_dict = history.history

# Plot training & validation accuracy values
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
