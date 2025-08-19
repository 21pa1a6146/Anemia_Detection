from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2 as cv
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
MODEL_PATH = os.path.join(os.getcwd(), 'anemia.keras')
model = load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image_resized = cv.resize(image, (64, 64))  # Resize to model input size
    image_normalized = image_resized / 255.0    # Normalize pixel values
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Precautionary measures for anemia
def get_anemia_precautions():
    return [
        "Consume a diet rich in iron (e.g., spinach, lentils, red meat).",
        "Include vitamin C in your meals to enhance iron absorption.",
        "Stay hydrated by drinking plenty of water.",
        "Avoid tea or coffee immediately after meals as it may hinder iron absorption.",
        "Consider iron supplements if prescribed by a doctor.",
        "Schedule regular checkups to monitor hemoglobin levels.",
        "Include folic acid and vitamin B12 in your diet (e.g., eggs, dairy, fortified cereals)."
    ]

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image and make predictions
        image = Image.open(file_path)
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        result = "Anemia" if class_idx == 0 else "Not Anemia"
        precautions = get_anemia_precautions() if result == "Anemia" else []

        # Use url_for to dynamically fetch the image URL
        return render_template(
            'index.html',
            result=result,
            confidence=f"{confidence:.2f}",
            precautions=precautions,
            image_url=url_for('static', filename=f'uploads/{filename}')
        )

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
