# AI-Powered Handwriting Recognition

A transformer-based OCR system that converts handwritten notes/images into digital text using Microsoft's TrOCR model.

## Features

- ✨ **Transformer-based OCR** - Uses TrOCR (Vision Encoder-Decoder model)
- 📊 **Evaluation Metrics** - Calculates CER (Character Error Rate) and WER (Word Error Rate)
- 🖼️ **Image Preprocessing** - Automatic grayscale conversion and normalization
- 🌐 **Web Interface** - User-friendly Flask web application
- ⚡ **GPU Support** - Automatic CUDA detection for faster inference

## Project Structure

```
handwriting_recognition/
├── app.py                  # Main Flask application
├── download_model.py       # Script to download TrOCR model
├── manual_download.py      # Alternative manual download script
├── test_prediction.py      # Test script for predictions
├── requirements.txt        # Python dependencies
├── models/
│   └── trocr_base/        # Pre-trained TrOCR model (local)
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded images directory
└── utils/
    └── metrics.py         # CER & WER calculation functions
```

## Installation

### 1. Clone the repository
```bash
cd d:\iforge_project\handwriting_recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the TrOCR model (if not already present)
```bash
python download_model.py
```

## Evaluation on IAM Dataset

To evaluate the model on the IAM Handwriting Database:

1.  Ensure `TrOCR/iam_words/words.txt` exists.
2.  Place the IAM word images in `TrOCR/iam_words/words/` (maintaining the IAM directory structure `a01/a01-000u/...`).
3.  Run the evaluation script:
    ```bash
    python evaluate_iam.py
    ```
    This script will load the dataset, preprocess images (grayscale + normalize), run inference, and calculate average CER and WER.

## Usage

### Run the Web Application
```bash
python app.py
```

The application will start on `http://localhost:5000`.

### Using the Web Interface
1.  Open your browser and navigate to `http://localhost:5000`.
2.  Click "Choose File" and select a handwritten image.
3.  (Optional) Enter ground truth text to calculate CER/WER metrics.
4.  Click "Recognize Text" to see the prediction.

### Test Prediction Script
```bash
python test_prediction.py
```

## Model Information

- **Model**: Microsoft TrOCR Base Handwritten
- **Architecture**: Vision Encoder-Decoder (ViT + RoBERTa)
- **Input**: RGB images of handwritten text
- **Output**: Recognized text string

## Evaluation Metrics

- **CER (Character Error Rate)**: Measures character-level accuracy
- **WER (Word Error Rate)**: Measures word-level accuracy

Lower values indicate better performance (0.0 = perfect match)

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended - FREE)
1. Create account at https://huggingface.co/spaces
2. Create new Space with Gradio/Streamlit
3. Upload files and deploy

### Option 2: Render (Free Tier)
1. Create account at https://render.com
2. Connect GitHub repository
3. Deploy as Web Service

### Option 3: Railway (Free Tier)
1. Create account at https://railway.app
2. Deploy from GitHub
3. Automatic environment setup

### Option 4: Streamlit Cloud (Free)
1. Convert Flask app to Streamlit (optional)
2. Deploy at https://streamlit.io/cloud

### Option 5: Local with Ngrok
```bash
pip install pyngrok
ngrok http 5000
```

## Requirements

- Python 3.9+
- PyTorch 2.x
- Transformers 5.x
- Flask 2.x
- 2GB+ RAM (4GB+ recommended)
- GPU optional (CUDA compatible)

## Troubleshooting

### Model not loading?
Run: `python download_model.py` to re-download the model

### CUDA out of memory?
The app automatically falls back to CPU if GPU is not available

### Import errors?
Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

MIT License - Feel free to use for personal and commercial projects

## Credits

- **TrOCR Model**: Microsoft Research
- **Framework**: Flask, PyTorch, Hugging Face Transformers
- **Evaluation**: jiwer library for CER/WER metrics
