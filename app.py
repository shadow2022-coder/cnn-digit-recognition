import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
print("Loading CNN model...")
try:
    model = keras.models.load_model('cnn_model.h5')
    print("✓ Model loaded!")
except Exception as e:
    print(f"Error: {e}")
    model = None

def predict_digit(canvas_data):
    """
    Predict digit from canvas drawing
    canvas_data: dict with 'composite' key containing the drawn image
    """
    try:
        if canvas_data is None:
            return "Draw something first!", {}, "No image"
        
        # Get the image from canvas
        if isinstance(canvas_data, dict):
            canvas_image = canvas_data['composite']
        else:
            canvas_image = canvas_data
        
        if canvas_image is None:
            return "Draw something first!", {}, "No image"
        
        # Convert to PIL Image
        image = Image.fromarray(canvas_image.astype('uint8')).convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Preprocess
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = model.predict(image_array, verbose=0)[0]
        
        # Get results
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit] * 100)
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = {
            f"Digit {int(idx)}": float(predictions[idx] * 100)
            for idx in top_3_indices
        }
        
        result_text = f"🎯 Predicted Digit: **{predicted_digit}**\n\n📊 Confidence: **{confidence:.1f}%**"
        
        return result_text, top_predictions, f"✓ Prediction successful!"
    
    except Exception as e:
        return f"Error: {str(e)}", {}, "❌ Prediction failed"

# Create Gradio interface
def create_demo():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("""
        # 🧠 CNN Digit Recognition
        **Draw a handwritten digit (0-9) and let AI predict it!**
        
        This app uses a trained Convolutional Neural Network (CNN) with 98%+ accuracy.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Draw Here")
                canvas = gr.Sketchpad(
                    label="Canvas",
                    type="numpy",
                    shape=(280, 280),
                    scale=1
                )
                
                gr.Markdown("""
                #### Instructions:
                1. Draw a digit on the canvas
                2. AI predicts automatically
                3. Click Clear to draw again
                """)
                
                clear_btn = gr.Button("🗑️ Clear", size="lg")
                clear_btn.click(lambda: None, outputs=canvas)
            
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                
                result_text = gr.Markdown("👇 Draw something to get started!")
                
                gr.Markdown("#### Top Predictions")
                result_chart = gr.BarChart(
                    label="Confidence Scores (%)",
                    x_title="Digit",
                    y_title="Confidence (%)",
                    height=300
                )
                
                status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
        
        # Auto-predict on canvas change
        canvas.change(
            fn=predict_digit,
            inputs=canvas,
            outputs=[result_text, result_chart, status],
            show_progress=False
        )
    
    return demo

# Launch
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
