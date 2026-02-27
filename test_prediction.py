import os
import sys
from app import predict_image

# Path to a test image
test_image_path = r"d:\iforge_project\handwriting_recognition\TrOCR\Scripts\trocr_image.jpg"

if not os.path.exists(test_image_path):
    print(f"Test image not found at {test_image_path}")
else:
    print(f"Testing prediction on {test_image_path}...")
    try:
        text = predict_image(test_image_path)
        print(f"Predicted Text: {text}")
    except Exception as e:
        print(f"Error: {e}")
