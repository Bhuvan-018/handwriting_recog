import os
import torch
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from jiwer import cer, wer
from utils.preprocessing import preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model Configuration
# Use locally downloaded model
MODEL_PATH = "models/trocr_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

def load_model():
    global processor, model
    try:
        # Load from local directory
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        processor = None
        model = None

load_model()

def predict_image(image_path):
    if model is None or processor is None:
        return "Model not loaded."
    
    try:
        # Use preprocessing utility for consistent handling
        # It handles grayscale conversion and normalization via processor
        pixel_values = preprocess_image(image_path, processor)
        if pixel_values is None:
             return "Error preprocessing image."
             
        pixel_values = pixel_values.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return generated_text
    except Exception as e:
        return f"Error during prediction: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    metrics = None
    uploaded_image_name = None
    ground_truth = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        ground_truth = request.form.get('ground_truth', '').strip()
        
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image_name = filename
            
            # Run Inference
            prediction = predict_image(filepath)
            
            # Calculate Metrics if Ground Truth is provided
            if ground_truth:
                try:
                    cer_score = cer(ground_truth, prediction)
                    wer_score = wer(ground_truth, prediction)
                    metrics = {
                        'CER': f"{cer_score:.4f}",
                        'WER': f"{wer_score:.4f}"
                    }
                except Exception as e:
                    metrics = {'Error': str(e)}
    
    return render_template('index.html', 
                           prediction=prediction, 
                           metrics=metrics, 
                           uploaded_image=uploaded_image_name,
                           ground_truth=ground_truth)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
