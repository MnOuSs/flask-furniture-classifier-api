"""
model_train.py

This module builds, trains, and saves a Convolutional Neural Network (CNN)
for classifying furniture images into categories such as bed, chair, sofa,
swivelchair, and table.

Workflow:
1. Load and preprocess the image datasets for training and validation.
2. Normalize the image pixel values for better model performance.
3. Define and compile a CNN architecture using Keras.
4. Train the model for 45 epochs on the training data.
5. Save the trained model as an .h5 file for later prediction or deployment.
"""
import os
import json
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Path to the training data and validation data directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "images/train")
VAL_DIR = os.path.join(BASE_DIR, "images/val")

# Image dimensions and batch size for loading
img_height, img_width = 128, 128
BATCH_SIZE = 32

# Load dataset from the directory structure
# Each subfolder is treated as a class label (e.g., bed, chair, sofa, etc.)
train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
)

val_dataset = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
)

# Retrieve the class names from folder names
class_names = train_dataset.class_names

# Save class names to a JSON file for later reference during prediction
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f)

# Normalize pixel values (0–255 → 0–1)
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Build the Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=45,
)

# Save the trained model to an .h5 file
model.save("furniture_classification_model.h5")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('training_validation_accuracy.png', dpi=300)
plt.show()
