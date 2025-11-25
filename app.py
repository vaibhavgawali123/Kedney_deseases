from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model once when the app starts
model = load_model("model_converted.h5")
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or not allowed_file(request.files['image'].filename):
        return jsonify({'error': 'No image uploaded or invalid file format'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        prediction, confidence = predict_image(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)  # Clean up the uploaded file

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")

