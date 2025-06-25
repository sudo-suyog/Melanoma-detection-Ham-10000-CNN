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
    
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model1

# Define the tuner
tuner = BayesianOptimization(
    create_alexnet,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='my_dir',
    project_name='ham10000_alexnet_tuning',overwrite=True)

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=1, validation_split=0.2)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Re-instantiate the model with the best hyperparameters and train it on the entire training set
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=20, validation_split=0.2)
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Train the model and capture the history
history = best_model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# Get training history
history_dict = history.history

# Evaluate the model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)

# Plotting accuracy
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
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

# Predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute confusion matrix
confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred_classes[i]] += 1


def calculate_sensitivity_specificity(confusion_matrix):
    sensitivity = np.zeros(len(classes))
    specificity = np.zeros(len(classes))
    for i in range(len(classes)):
        true_positives = confusion_matrix[i, i]
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        true_negatives = np.sum(confusion_matrix[np.arange(len(classes)) != i, :][:, np.arange(len(classes)) != i])
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        sensitivity[i] = true_positives / (true_positives + false_negatives)
        specificity[i] = true_negatives / (true_negatives + false_positives)
    return sensitivity, specificity

sensitivity, specificity = calculate_sensitivity_specificity(confusion_matrix)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Plot sensitivity and specificity
plt.figure(figsize=(10, 6))

# Plot sensitivity
plt.plot(classes, sensitivity, marker='o', linestyle='-', color='b', label='Sensitivity')

# Plot specificity
plt.plot(classes, specificity, marker='o', linestyle='-', color='r', label='Specificity')

plt.xlabel('Classes')
plt.ylabel('Metric Value')
plt.title('Sensitivity and Specificity by Class')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.show()

from sklearn.metrics import roc_curve, auc
# Plot ROC curve
plt.figure(figsize=(10, 5))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



