from PIL import Image
import torch
from transformers import TrOCRProcessor

def preprocess_image(image_path, processor):
    """
    Preprocess image for TrOCR model.
    Converts to grayscale (L mode) then to RGB (since TrOCR expects 3 channels).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        # Although TrOCR is trained on RGB images, for handwriting recognition,
        # converting to grayscale and back to RGB can help normalize color variations.
        # But if the user specifically asked for grayscale, we should respect that.
        # However, TrOCR preprocessor expects 3 channels usually.
        # So we convert to 'L' then back to 'RGB' to simulate grayscale input.
        image = image.convert("L").convert("RGB")
        
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return pixel_values
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None
