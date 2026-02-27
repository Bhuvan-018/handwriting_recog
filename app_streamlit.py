import streamlit as st
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from jiwer import cer, wer
import os

# Page config
st.set_page_config(
    page_title="Handwriting Recognition",
    page_icon="✍️",
    layout="wide"
)

# Model Configuration
MODEL_PATH = "models/trocr_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    """Load the TrOCR model and processor"""
    try:
        if os.path.exists(MODEL_PATH):
            processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
            return processor, model, None
        else:
            # Try loading from Hugging Face Hub
            model_name = "microsoft/trocr-base-handwritten"
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
            return processor, model, None
    except Exception as e:
        return None, None, str(e)

def predict_image(image, processor, model):
    """Run OCR prediction on image"""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return generated_text, None
    except Exception as e:
        return None, str(e)

# Main App
def main():
    st.title("✍️ AI-Powered Handwriting Recognition")
    st.markdown("**Powered by Microsoft TrOCR (Transformer-based OCR)**")
    
    # Load model
    with st.spinner("Loading TrOCR model..."):
        processor, model, error = load_model()
    
    if error:
        st.error(f"❌ Model loading failed: {error}")
        st.info("💡 Try running `python download_model.py` to download the model first.")
        return
    
    st.success(f"✅ Model loaded successfully on {DEVICE.upper()}")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses **TrOCR** (Transformer-based OCR) 
        to recognize handwritten text from images.
        
        **Features:**
        - 🖼️ Upload handwritten images
        - 📝 Get digital text output
        - 📊 Calculate CER & WER metrics
        - ⚡ GPU-accelerated (if available)
        """)
        
        st.header("📊 Metrics")
        st.markdown("""
        - **CER**: Character Error Rate
        - **WER**: Word Error Rate
        
        Lower values = better accuracy
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a handwritten image...", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image containing handwritten text"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Ground truth input
            ground_truth = st.text_input(
                "Ground Truth (optional)",
                help="Enter the expected text to calculate CER/WER metrics"
            )
    
    with col2:
        st.subheader("📝 Recognition Result")
        
        if uploaded_file is not None:
            if st.button("🔍 Recognize Text", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    prediction, error = predict_image(image, processor, model)
                
                if error:
                    st.error(f"❌ Prediction failed: {error}")
                else:
                    st.success("✅ Recognition complete!")
                    
                    # Display prediction
                    st.markdown("**Predicted Text:**")
                    st.code(prediction, language=None)
                    
                    # Calculate metrics if ground truth provided
                    if ground_truth and ground_truth.strip():
                        try:
                            cer_score = cer(ground_truth, prediction)
                            wer_score = wer(ground_truth, prediction)
                            
                            st.markdown("---")
                            st.markdown("**📊 Evaluation Metrics:**")
                            
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("CER", f"{cer_score:.4f}")
                            with metric_col2:
                                st.metric("WER", f"{wer_score:.4f}")
                                
                        except Exception as e:
                            st.warning(f"⚠️ Metrics calculation failed: {e}")
        else:
            st.info("👈 Please upload an image to begin recognition")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit • TrOCR • PyTorch • Transformers</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
