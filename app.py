"""
app.py

Flask REST API for furniture classification.
Runs locally and is ready for cloud deployment.
"""
import os
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import keras
from keras.utils import load_img, img_to_array

app = Flask(__name__)
CORS(app)

# Load model and class names
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'furniture_classification_model.h5')
CLASS_FILE = os.path.join(BASE_DIR, 'class_names.json')

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_FILE, 'r', encoding="utf-8") as f:
    class_names = json.load(f)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    '''Receives an image via POST and returns predicted class and the confidence.'''
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]

    try:
        img = load_img(io.BytesIO(file.read()), target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[int(np.argmax(predictions))]
        confidence = round(100 * float(np.max(predictions)), 2)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence}%"
        })
    except (OSError, ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
