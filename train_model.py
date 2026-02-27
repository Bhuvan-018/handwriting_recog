import os
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from PIL import Image
from sklearn.model_selection import train_test_split
from jiwer import cer
import numpy as np
from utils.dataset import load_iam_words_dataset

class IAMDataset(Dataset):
    def __init__(self, dataset, processor, max_target_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, text = self.dataset[idx]
        
        # Open and convert image
        # Using .convert("RGB") because TrOCR expects RGB images
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        
        # Important: make sure we don't compute loss on padding tokens
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer_score = cer(label_str, pred_str)

    return {"cer": cer_score}

if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths for Colab environment if needed
    DATASET_DIR = "TrOCR/iam_words"  # Should be relative to where you run the script
    MODEL_NAME = "microsoft/trocr-base-handwritten" # Start fine-tuning from this base
    OUTPUT_DIR = "models/trocr_finetuned_iam"
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(DATASET_DIR, "words.txt")):
        print(f"Error: Dataset not found at {DATASET_DIR}")
        print("Please ensure you have uploaded the IAM words dataset.")
        exit(1)
        
    # --- Load Data ---
    print(f"Loading dataset from {DATASET_DIR}...")
    full_dataset = load_iam_words_dataset(DATASET_DIR)
    
    if not full_dataset:
        print("No data found!")
        exit(1)
        
    print(f"Found {len(full_dataset)} samples.")
    
    # Split into train and test
    train_data, test_data = train_test_split(full_dataset, test_size=0.1, random_state=42)
    print(f"Training on {len(train_data)} samples, Evaluating on {len(test_data)} samples.")

    # --- Initialize Processor and Model ---
    print(f"Loading model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Make sure vocab size is correct
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Create Datasets
    train_dataset = IAMDataset(train_data, processor)
    eval_dataset = IAMDataset(test_data, processor)

    # --- Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        output_dir=OUTPUT_DIR,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        num_train_epochs=3, # Adjust epochs as needed
        report_to="none", # Disable wandb/tensorboard for simple runs
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor, # Pass feature extractor as tokenizer for collator
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.train()

    # --- Save Final Model ---
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Training complete and model saved.")
