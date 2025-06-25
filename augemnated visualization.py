import os
import cv2
import numpy as np
import pandas as pd
import random
from tensorflow.keras.layers import MaxPooling2D, Dropout
from keras.layers import Conv2D, Flatten,Dense,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split

# Load metadata
metadata_path = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\HAM10000_metadata.csv'  
metadata_df = pd.read_csv(metadata_path)

data_dir = r'C:\Users\DELL\Desktop\Autmn(2024)\Neural Nets (Assign)\HAM10000 dataset\reorganized'
classes = ['akiec', 'bcc', 'bkl', 'df','mel', 'nv', 'vasc']
# Defining the Image size
Image_Size = 224
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

# Scaling down the pixel
#images_data = images_data / 255

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt

# Define the data generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images and visualize
augmented_gen = datagen.flow(X_train, y_train, batch_size=32, shuffle=False)

# Generate and visualize the first batch of augmented images
# Generate and inspect the first batch of augmented images
for batch_X, batch_y in augmented_gen:
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Plotting the first 9 augmented images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(batch_X[i].astype('uint8'))  # Convert to unsigned integer for imshow
        plt.axis("off")
    plt.show()
    print(batch_X[0])  # Print the first augmented image to check its contents
    break
