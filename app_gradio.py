import gradio as gr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from jiwer import cer, wer
import os

# Model Configuration
MODEL_PATH = "bhuvan-018/trocr-handwriting-iam"
BASE_MODEL_PATH = "models/trocr_base"
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")

# Load model
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading fine-tuned model from {MODEL_PATH}")
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    elif os.path.exists(BASE_MODEL_PATH):
        print(f"Loading base model from {BASE_MODEL_PATH}")
        processor = TrOCRProcessor.from_pretrained(BASE_MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    else:
        # Load from Hugging Face Hub
        print(f"Loading default model from Hub: {DEFAULT_MODEL}")
        processor = TrOCRProcessor.from_pretrained(DEFAULT_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_MODEL).to(DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    processor = None
    model = None

def recognize_handwriting(image, ground_truth=""):
    """
    Recognize handwritten text from image
    """
    if model is None or processor is None:
        return "Error: Model not loaded", ""
    
    try:
        # Convert numpy array to PIL Image if needed (Gradio usually returns PIL or numpy)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
            
        # Run inference
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate metrics if ground truth is provided
        metrics_text = ""
        if ground_truth and ground_truth.strip():
            try:
                cer_score = cer(ground_truth, prediction)
                wer_score = wer(ground_truth, prediction)
                metrics_text = f"**Evaluation Metrics:**\n\n📊 CER: {cer_score:.4f}\n📊 WER: {wer_score:.4f}"
            except Exception as e:
                metrics_text = f"Metrics calculation failed: {e}"
        
        return prediction, metrics_text
        
    except Exception as e:
        return f"Error: {e}", ""

# Create Gradio interface
with gr.Blocks(title="Handwriting Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✍️ AI-Powered Handwriting Recognition
        **Powered by Microsoft TrOCR (Transformer-based OCR)**
        
        Upload a handwritten image and get digital text output. Optionally provide ground truth text to calculate accuracy metrics.
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Handwritten Image")
            ground_truth_input = gr.Textbox(
                label="Ground Truth (Optional)", 
                placeholder="Enter expected text for evaluation",
                lines=2
            )
            submit_btn = gr.Button("🔍 Recognize Text", variant="primary")
        
        with gr.Column():
            prediction_output = gr.Textbox(label="Predicted Text", lines=5)
            metrics_output = gr.Markdown(label="Metrics")
    
    gr.Markdown(
        """
        ---
        **Features:**
        - 🖼️ Transformer-based OCR using Microsoft TrOCR
        - 📊 Character Error Rate (CER) and Word Error Rate (WER) evaluation
        - ⚡ GPU-accelerated inference when available
        - 🎯 Optimized for handwritten text recognition
        
        **Model:** `microsoft/trocr-base-handwritten` (Fine-tuned on IAM)
        """
    )
    
    # Connect button to function
    submit_btn.click(
        fn=recognize_handwriting,
        inputs=[image_input, ground_truth_input],
        outputs=[prediction_output, metrics_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
