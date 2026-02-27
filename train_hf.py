import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from datasets import load_dataset
from jiwer import cer
import numpy as np

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer_score = cer(label_str, pred_str)

    return {"cer": cer_score}

def preprocess_function(examples):
    # Retrieve the image and text from the batch
    # The dataset columns are likely 'image' and 'text'
    # Teklia/IAM-line usually has 'image' and 'text' columns
    
    images = examples['image']
    texts = examples['text']
    
    # Preprocess images
    # Ensure images are RGB
    images = [image.convert("RGB") for image in images]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    
    # Tokenize text
    labels = processor.tokenizer(
        texts, 
        padding="max_length", 
        max_length=128
    ).input_ids
    
    # Replace padding token id with -100 to ignore in loss
    labels = [
        [label if label != processor.tokenizer.pad_token_id else -100 for label in label_seq]
        for label_seq in labels
    ]
    
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    # --- Configuration ---
    # Using the Teklia/IAM-line dataset from Hugging Face
    # This dataset contains line-level images which are better for TrOCR context
    DATASET_ID = "Teklia/IAM-line" 
    MODEL_NAME = "microsoft/trocr-base-handwritten"
    OUTPUT_DIR = "models/trocr_finetuned_iam_hf"
    
    # --- Load Dataset from Hugging Face ---
    print(f"Loading dataset: {DATASET_ID}...")
    try:
        dataset = load_dataset(DATASET_ID)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
        
    print(f"Dataset structure: {dataset}")
    
    # --- Initialize Processor and Model ---
    print(f"Loading model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # --- Preprocess Dataset ---
    print("Preprocessing dataset...")
    # Map the preprocessing function over the dataset
    # We remove the original columns to format it for the trainer
    column_names = dataset["train"].column_names
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        batch_size=8
    )
    
    # Use standard splits if available, otherwise create them
    if "train" in tokenized_dataset:
        train_dataset = tokenized_dataset["train"]
    else:
        # Fallback if structure is different
        train_dataset = tokenized_dataset
        
    if "validation" in tokenized_dataset:
        eval_dataset = tokenized_dataset["validation"]
    elif "test" in tokenized_dataset:
        eval_dataset = tokenized_dataset["test"]
    else:
        # If no validation/test split, split the train dataset
        split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"Training on {len(train_dataset)} samples")
    print(f"Evaluating on {len(eval_dataset)} samples")

    # --- Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=torch.cuda.is_available(),
        output_dir=OUTPUT_DIR,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        num_train_epochs=3,
        report_to="none",
        remove_unused_columns=False # Important for custom dataset columns if any
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
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
