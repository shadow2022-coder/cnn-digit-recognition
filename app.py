from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='.', static_folder='.')

# Load model ONCE at startup
print("Loading trained model...")
try:
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = keras.models.load_model('cnn_model.h5')
    print("✓ Model loaded successfully!")
    model_ready = True
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    model_ready = False

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict digit from canvas image - OPTIMIZED"""
    try:
        if not model_ready:
            return jsonify({'error': 'Model not ready'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode image (FAST)
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Preprocess (FAST)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Predict (with timeout protection)
        try:
            predictions = model.predict(image_array, verbose=0)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Get results (FAST)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit] * 100)
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        all_predictions = {
            str(int(idx)): float(predictions[0][idx] * 100)
            for idx in top_3_indices
        }
        
        return jsonify({
            'status': 'success',
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'loaded' if model_ready else 'failed'}), 200

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧠 CNN DIGIT RECOGNITION")
    print("="*70)
    app.run(debug=False, host='0.0.0.0', port=5000)
