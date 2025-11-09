from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model
print("Loading trained model...")
try:
    model = keras.models.load_model('model/cnn_model.h5')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict digit from canvas image"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit]) * 100
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_indices] * 100
        
        all_predictions = {
            str(int(top_3_indices[i])): float(top_3_probs[i])
            for i in range(len(top_3_indices))
        }
        
        return jsonify({
            'status': 'success',
            'digit': int(predicted_digit),
            'confidence': float(confidence),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧠 CNN DIGIT RECOGNITION - LOCAL APP")
    print("="*70)
    print("Opening at: http://localhost:5000")
    print("="*70)
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5000)
