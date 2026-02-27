import os
import requests
import json

MODEL_NAME = "microsoft/trocr-base-handwritten"
SAVE_DIR = "models/trocr_base"
os.makedirs(SAVE_DIR, exist_ok=True)

files_to_download = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "generation_config.json"
]

base_url = f"https://huggingface.co/{MODEL_NAME}/resolve/main/"

print(f"Manually downloading model files for {MODEL_NAME} to {SAVE_DIR}...")

for filename in files_to_download:
    url = base_url + filename
    save_path = os.path.join(SAVE_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"File {filename} already exists. Skipping.")
        continue
        
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("Manual download complete.")

# Verify and load
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    print("Verifying loaded model...")
    processor = TrOCRProcessor.from_pretrained(SAVE_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(SAVE_DIR)
    print("Verification successful!")
except Exception as e:
    print(f"Verification failed: {e}")
