import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils.dataset import load_iam_words_dataset
from utils.preprocessing import preprocess_image
from jiwer import cer, wer
import os
from tqdm import tqdm

def evaluate_on_iam(dataset_path="TrOCR/iam_words", model_path="models/trocr_base"):
    """
    Evaluates the TrOCR model on the IAM Words dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_iam_words_dataset(dataset_path)
    if not dataset:
        print("No data found or empty dataset.")
        return
    
    print(f"Found {len(dataset)} samples.")

    total_cer = 0.0
    total_wer = 0.0
    count = 0

    print("Starting evaluation...")
    for image_path, ground_truth in tqdm(dataset):
        if not os.path.exists(image_path):
            continue
            
        try:
            pixel_values = preprocess_image(image_path, processor)
            if pixel_values is None:
                continue
                
            pixel_values = pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate metrics
            current_cer = cer(ground_truth, generated_text)
            current_wer = wer(ground_truth, generated_text)
            
            total_cer += current_cer
            total_wer += current_wer
            count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    if count > 0:
        avg_cer = total_cer / count
        avg_wer = total_wer / count
        print(f"\nEvaluation Results on {count} samples:")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average WER: {avg_wer:.4f}")
    else:
        print("No samples were processed successfully.")

if __name__ == "__main__":
    evaluate_on_iam()
