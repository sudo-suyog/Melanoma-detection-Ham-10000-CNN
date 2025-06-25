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

#Designing individual models for image classification
#Alex net architecture


def create_alexnet(num_classes):
    model1 = Sequential()
    model1.add(BatchNormalization())
    # Layer 1
    model1.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 2
    model1.add(Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model1.add(BatchNormalization())
    # Layer 3
    model1.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
    model1.add(BatchNormalization())
    # Layer 4

    model1.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
    model1.add(BatchNormalization())
    # Layer 5
    model1.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model1.add(BatchNormalization())
    # Flatten
    model1.add(Flatten())
        # Layer 6
    model1.add(Dense(4096, activation='relu'))
    model1.add(Dropout(0.5))
    # Layer 7
    model1.add(Dense(4096, activation='relu'))
    model1.add(Dropout(0.5))
        # Output layer
    model1.add(Dense(num_classes, activation='softmax')
          )    
    return model1

#def create_vgg16(num_classes):
    #model2 = Sequential()

    # Block 1
    #model2.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    #model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    #model2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    #model2.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully Connected layers
    #model2.add(Flatten())
    #model2.add(Dense(4096, activation='relu'))
    #model2.add(Dropout(0.5))
    #model2.add(Dense(4096, activation='relu'))
    #model2.add(Dropout(0.5))
    #model2.add(Dense(num_classes, activation='softmax'))

    #return model2
num_classes=len(classes)



# defining individual model 
#vgg16_model2 =create_vgg16(num_classes)
#vgg16_model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

alexnet_model1 = create_alexnet(num_classes)
# Compile the model
alexnet_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history1 = alexnet_model1.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)#history2 = vgg16_model2.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

# Evaluate the model on the test data
#test_loss, test_acc = vgg16_model2.evaluate(X_test, y_test)
#print("Test Accuracy of vgg16:", test_acc)
#test_loss, test_acc = alexnet_model1.evaluate(X_test, y_test)
#print("Test Accuracy:", test_acc)



