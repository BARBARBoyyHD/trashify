import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your saved model
model_path = os.path.join('./models', 'ModelSampah.h5')  # Ensure this path is correct
model = load_model(model_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp','svg'}  # Allowed image extensions

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess the uploaded image to match model input requirements."""
    # Convert the image to RGB (in case it's in grayscale or other format)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((128, 128))  # Resize to the expected input size (128x128)
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    
    # Ensure the image has shape (1, 128, 128, 3) by adding the batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Hello, World!"})


@app.route('/api/predict', methods=['POST'])
def predict():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed, upload an image with extension: png, jpg, jpeg, or gif"}), 400

    try:
        # Verify that the file content is an image
        image = Image.open(file.stream)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions using the loaded model
        predictions = model.predict(processed_image)

        # Assuming binary classification (Organik/Anorganik)
        class_names = ['Organik', 'Anorganik']  # Adjust to your model's labels
        predicted_class = class_names[np.argmax(predictions[0])]

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(np.max(predictions[0]))
        })

    except Exception as e:
        return jsonify({"error": f"Invalid image content: {str(e)}"}), 400


if __name__ == '__main__':
    app.run(debug=True)
