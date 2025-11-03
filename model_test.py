"""
model_test.py

This script loads the trained CNN model for furniture classification
and allows the user to enter the path to any image for prediction.

Steps performed:
1. Load the trained model from disk.
2. Ask the user to provide the path to an image file.
3. Preprocess the image (resize and normalize).
4. Run the model to predict the furniture type and confidence score.
"""
import os
import keras
from keras.utils import load_img, img_to_array
import numpy as np

# Get the directory where the model is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "furniture_classification_model.h5")

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Class names (ensure this matches the order from training)
class_names = ['Bed', 'Chair', 'Sofa', 'Swivelchair', 'Table']

# Ask user for image path
img_path = input("Enter the path to the furniture image: ").strip()

# Try to load and predict
try:
    # 1. Load and resize the image
    img = load_img(img_path, target_size=(128, 128))
   
    # 2. Convert to array and normalize
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 3. Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    # 4. Display result
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence}%")

# Handle file not found or invalid image format
except (FileNotFoundError, OSError) as e:
    print("Please check that the image path and format are correct.")
