from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image

# Enable CORS
app = Flask(__name__)
CORS(app)

MODEL_PATH = "VecTorium.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = 150
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    img = cv2.imread(temp_path)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)

    predictions = model.predict([img_array, img_array, img_array])


    predicted_label = LABELS[np.argmax(predictions)]
    prediction_percentages = (predictions[0] * 100).round(2)


    _, buffer = cv2.imencode('.jpg', img_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')


    response = {
        "predicted_label": predicted_label,
        "prediction_probabilities": {LABELS[i]: float(prediction_percentages[i]) for i in range(len(LABELS))},
        "preprocessed_image": img_base64
    }


    os.remove(temp_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
