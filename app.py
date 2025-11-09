import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import base64

# Load model
model = keras.models.load_model('cnn_model.h5')

def predict_digit(canvas_image):
    """Predict from canvas drawing"""
    if canvas_image is None:
        return "Please draw something!", {}
    
    # Convert to proper format
    image = Image.fromarray(canvas_image['composite']).convert('L')
    image = image.resize((28, 28))
    
    # Preprocess
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = model.predict(image_array, verbose=0)
    digit = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][digit] * 100)
    
    # Top predictions
    top_3 = np.argsort(predictions[0])[-3:][::-1]
    result_dict = {f"Digit {int(idx)}": float(predictions[0][idx] * 100) for idx in top_3}
    
    return f"Predicted: {digit} ({confidence:.1f}%)", result_dict

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 CNN Digit Recognition")
    gr.Markdown("Draw a digit (0-9) and let AI predict it!")
    
    with gr.Row():
        canvas = gr.Sketchpad(
            label="Draw a digit",
            type="numpy",
            shape=(280, 280)
        )
        output = gr.Textbox(label="Prediction")
    
    chart = gr.BarChart(label="Confidence Scores")
    
    canvas.change(
        fn=predict_digit,
        inputs=canvas,
        outputs=[output, chart]
    )

if __name__ == "__main__":
    demo.launch()
