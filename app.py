import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model exactly as in predict.py
try:
    model = tf.keras.models.load_model("wall_crack_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and model:
        try:
            # Replicating the AI logic from predict.py
            img = Image.open(file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            pred = model.predict(img_array)[0][0]
            
            # Formatting exactly as predict.py logic
            score = float(pred)
            if pred > 0.5:
                result = "No Crack"
            else:
                result = "Crack Detected"
                
            return jsonify({
                'prediction': result,
                'score': score
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Model not loaded or invalid file'}), 500

if __name__ == '__main__':
    # Run the app locally
    app.run(debug=True, host='0.0.0.0', port=5000)
