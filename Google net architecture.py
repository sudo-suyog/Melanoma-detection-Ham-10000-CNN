import os
import cv2
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, concatenate, AveragePooling2D

from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, Flatten,Dense,BatchNormalization
from sklearn.model_selection import train_test_split

# Load metadata
metadata_path = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\HAM10000_metadata.csv'  
metadata_df = pd.read_csv(metadata_path)

data_dir = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\reorganized'
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
#Defining the Image size
Image_Size = 224
#dividing this in individual classes with 500 sample data to avoid overfiting
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
# thr image value is in X_images and y has output labels
# Check the shapes
print("X_images shape:", X_images.shape)
print("X_metadata shape:", X_metadata.shape)
print("y shape:", y.shape)

# Scaling down the pixel
X_images=X_images/255

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

def inception_module(x, filters):
    tower_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(filters[2], (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(filters[3], (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(filters[4], (1, 1), padding='same', activation='relu')(tower_3)

    tower_4 = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(x)

    return concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

def create_googlenet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # Stem
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Global Average Pooling
    x = AveragePooling2D((7, 7), padding='same')(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Define input shape and number of classes
input_shape = (Image_Size, Image_Size, 3)  # Assuming 224x224 RGB images
num_classes = len(classes)

# Create GoogleNet model
googlenet_model = create_googlenet(input_shape, num_classes)

# Compile the model
googlenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history_googlenet = googlenet_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
