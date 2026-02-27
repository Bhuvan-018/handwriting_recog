import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"
SAVE_DIR = "models/trocr_base"

print(f"Downloading model {MODEL_NAME} to {SAVE_DIR}...")
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("Downloading processor...")
    # Try with use_fast=False if the fast tokenizer is the issue
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Standard processor load failed ({e}), trying use_fast=False...")
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
        
    processor.save_pretrained(SAVE_DIR)
    print("Processor saved.")

    print("Downloading model...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print("Model saved.")
    
    print("Success! Model downloaded locally.")

except Exception as e:
    print(f"FAILED to download model: {e}")
    import traceback
    traceback.print_exc()
