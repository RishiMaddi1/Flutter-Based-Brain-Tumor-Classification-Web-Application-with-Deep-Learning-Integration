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

# Load the saved model
MODEL_PATH = "VecTorium.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size and labels
IMAGE_SIZE = 150
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Save the image temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # Preprocess the image
    img = cv2.imread(temp_path)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Pass the image three times for each model input
    predictions = model.predict([img_array, img_array, img_array])

    # Get the predicted label (index of the highest prediction)
    predicted_label = LABELS[np.argmax(predictions)]
    prediction_percentages = (predictions[0] * 100).round(2)

    # Convert preprocessed image to base64 for the response
    _, buffer = cv2.imencode('.jpg', img_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Format the response
    response = {
        "predicted_label": predicted_label,
        "prediction_probabilities": {LABELS[i]: float(prediction_percentages[i]) for i in range(len(LABELS))},
        "preprocessed_image": img_base64
    }

    # Remove the temporary file
    os.remove(temp_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
