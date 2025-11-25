from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Upload directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model once
MODEL_PATH = "model_converted.h5"
model = load_model(MODEL_PATH)

class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']


# Validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    preds = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    return predicted_class, confidence


# ---------- ROUTES ---------- #

@app.route("/")
def home():
    return jsonify({"message": "Kidney Disease Detection Backend Running Successfully!"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format (allowed: png, jpg, jpeg)'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        predicted_class, confidence = predict_image(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Remove file after prediction
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })


# ---------- MAIN ---------- #

if __name__ == '__main__':
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=False)
