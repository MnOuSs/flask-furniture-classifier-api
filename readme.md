# Furniture Image Classification API

This project is part of the **From Model to Production (DLBDSMTP01)** course in the University of Applied Sciences.  
It demonstrates how to train, package, and deploy a Convolutional Neural Network (CNN) as a RESTful API using **Flask** and **TensorFlow**.

---

## Overview

- **Goal:** Classify furniture images into 5 categories:
  `bed`, `chair`, `sofa`, `swivelchair`, and `table`.
- **Dataset:** Public furniture image dataset (sourced from Kaggle).
- **Model:** CNN built with Keras, trained for 45 epochs.
- **Deployment:** Flask API hosted on [Render](https://render.com).

---

## 🧠 System Architecture

User --> /predict (HTTP POST)
--> Flask API (app.py)
--> loads furniture_classification_model.h5
--> preprocesses uploaded image
--> returns JSON: {predicted_class, confidence}

---

## Local Setup

```bash
git clone https://github.com/MnOuSs/flask-furniture-classifier-api.git
pip install -r requirements.txt
python app.py
```
Then send a test request:

```bash
curl -X POST -F "file=@sample_images/chair.jpg" http://127.0.0.1:5000/predict
```

---

## Cloud API

Base URL:
https://flask-furniture-classifier-api.onrender.com

**Example**
```bash
curl -X POST -F "file=@sample_images/chair.jpg" https://flask-furniture-classifier-api.onrender.com/predict
```
**Result**
```bash
{"predicted_class": "chair", "confidence": "98.7%"}
```

## Technologies Involved

* Python3
* TensorFlow/Keras
* Flask & Flask-CORS
* Render (Cloud deployment)

## Author

Oussama Manai
IU International University of Applied Sciences
Course: DLBDSMTP01 – Project: From Model to Production