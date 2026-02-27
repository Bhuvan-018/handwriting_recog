"""
System Verification Script for Handwriting Recognition Project
Tests all components and dependencies
"""

import sys
import os

def print_status(test_name, passed, message=""):
    """Print colored status message"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"    {message}")

def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60)
    
    imports = {
        "Flask": "flask",
        "PyTorch": "torch",
        "Transformers": "transformers",
        "PIL/Pillow": "PIL",
        "jiwer": "jiwer",
        "requests": "requests"
    }
    
    all_passed = True
    for name, module in imports.items():
        try:
            __import__(module)
            print_status(name, True)
        except ImportError as e:
            print_status(name, False, str(e))
            all_passed = False
    
    return all_passed

def test_model_files():
    """Test if model files exist"""
    print("\n" + "="*60)
    print("Testing Model Files...")
    print("="*60)
    
    model_path = "models/trocr_base"
    required_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin"
    ]
    
    if not os.path.exists(model_path):
        print_status("Model directory", False, f"{model_path} not found")
        return False
    
    all_passed = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        print_status(file, exists, "" if exists else f"{file_path} not found")
        if not exists:
            all_passed = False
    
    return all_passed

def test_model_loading():
    """Test if model can be loaded"""
    print("\n" + "="*60)
    print("Testing Model Loading...")
    print("="*60)
    
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        MODEL_PATH = "models/trocr_base"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"    Device: {DEVICE}")
        
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
        print_status("Processor loaded", True)
        
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
        print_status("Model loaded", True)
        
        return True, processor, model, DEVICE
        
    except Exception as e:
        print_status("Model loading", False, str(e))
        return False, None, None, None

def test_prediction(processor, model, device):
    """Test prediction on a sample image"""
    print("\n" + "="*60)
    print("Testing Prediction...")
    print("="*60)
    
    try:
        from PIL import Image
        import torch
        
        # Check if test image exists
        test_image_path = "TrOCR/Scripts/trocr_image.jpg"
        if not os.path.exists(test_image_path):
            print_status("Test image", False, f"{test_image_path} not found")
            print("    Skipping prediction test")
            return False
        
        # Load and process image
        image = Image.open(test_image_path).convert("RGB")
        print_status("Image loaded", True, f"{test_image_path}")
        
        # Run prediction
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print_status("Prediction successful", True, f"Predicted: '{predicted_text}'")
        return True
        
    except Exception as e:
        print_status("Prediction", False, str(e))
        return False

def test_metrics():
    """Test CER and WER calculation"""
    print("\n" + "="*60)
    print("Testing Metrics...")
    print("="*60)
    
    try:
        from jiwer import cer, wer
        
        reference = "Hello World"
        hypothesis = "Hello World"
        
        cer_score = cer(reference, hypothesis)
        wer_score = wer(reference, hypothesis)
        
        print_status("CER calculation", True, f"CER: {cer_score:.4f}")
        print_status("WER calculation", True, f"WER: {wer_score:.4f}")
        
        return True
        
    except Exception as e:
        print_status("Metrics", False, str(e))
        return False

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\n" + "="*60)
    print("Testing Flask Application...")
    print("="*60)
    
    try:
        from app import app, predict_image
        print_status("Flask app import", True)
        print_status("predict_image function", True)
        return True
    except Exception as e:
        print_status("Flask app", False, str(e))
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n" + "="*60)
    print("Testing Directory Structure...")
    print("="*60)
    
    directories = [
        "models",
        "templates",
        "uploads",
        "utils"
    ]
    
    all_passed = True
    for directory in directories:
        exists = os.path.exists(directory)
        print_status(directory, exists)
        if not exists:
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "HANDWRITING RECOGNITION SYSTEM VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    
    # Test directories
    results['directories'] = test_directories()
    
    # Test model files
    results['model_files'] = test_model_files()
    
    # Test model loading
    model_loaded, processor, model, device = test_model_loading()
    results['model_loading'] = model_loaded
    
    # Test prediction (only if model loaded)
    if model_loaded:
        results['prediction'] = test_prediction(processor, model, device)
    else:
        results['prediction'] = False
    
    # Test metrics
    results['metrics'] = test_metrics()
    
    # Test Flask app
    results['flask_app'] = test_flask_app()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        print("\nNext steps:")
        print("  1. Run Flask app: python app.py")
        print("  2. Visit: http://localhost:5000")
        print("  3. Deploy using DEPLOYMENT.md guide")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        if not results['model_files']:
            print("\n💡 Tip: Run 'python download_model.py' to download the model")
        return 1

if __name__ == "__main__":
    sys.exit(main())
