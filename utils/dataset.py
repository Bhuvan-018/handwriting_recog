import os

def load_iam_words_dataset(dataset_dir):
    """
    Loads IAM Words dataset from the given directory.
    Expected structure:
        dataset_dir/words.txt
        dataset_dir/words/a01/a01-000u/a01-000u-00-00.png
    
    Returns a list of tuples: (image_path, text_label)
    """
    words_file = os.path.join(dataset_dir, "words.txt")
    if not os.path.exists(words_file):
        print(f"Warning: {words_file} not found.")
        return []

    dataset = []
    
    with open(words_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
            
        parts = line.strip().split()
        if len(parts) < 9:
            continue
            
        word_id = parts[0]
        result_status = parts[1]
        
        # Only use 'ok' status words if strict filtering is needed, 
        # but usually we use all non-error ones or just filter by status.
        if result_status == "err":
            continue
            
        # The text label starts from index 8 onwards (can contain spaces?)
        # Actually IAM words.txt format says the last part is the label.
        # But looking at example: "a01-000u-00-00 ok 154 1 408 768 27 51 AT A"
        # The label is the last part. Wait, 'AT' is likely the text? No 'A' is likely part of it?
        # Actually, the format is: word_id result_status gray_level number_of_components x y w h tag text
        # So text is the last part.
        text_label = parts[-1]
        
        # Construct image path
        # word_id format: a01-000u-00-00
        # folder1: a01
        # folder2: a01-000u
        folder1 = word_id.split("-")[0]
        folder2 = f"{folder1}-{word_id.split('-')[1]}"
        
        image_path = os.path.join(dataset_dir, "words", folder1, folder2, f"{word_id}.png")
        
        # Check for .jpg if .png not found (IAM is usually png but sometimes jpg)
        if not os.path.exists(image_path):
             image_path_jpg = image_path.replace(".png", ".jpg")
             if os.path.exists(image_path_jpg):
                 image_path = image_path_jpg
             else:
                 # If image doesn't exist, skip it or keep it but mark as missing?
                 # For now, skip to avoid errors during loading.
                 # But since user might not have downloaded images, we might return it anyway 
                 # so they know what's missing.
                 # Let's check if 'words' directory exists at all.
                 words_dir = os.path.join(dataset_dir, "words")
                 if not os.path.exists(words_dir):
                     # If the whole directory is missing, we probably shouldn't return anything or just return paths.
                     pass
        
        dataset.append((image_path, text_label))
        
    return dataset
