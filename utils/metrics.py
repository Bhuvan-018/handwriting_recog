from jiwer import cer, wer

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER).
    """
    if not reference:
        return 0.0
    return cer(reference, hypothesis)

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER).
    """
    if not reference:
        return 0.0
    return wer(reference, hypothesis)
