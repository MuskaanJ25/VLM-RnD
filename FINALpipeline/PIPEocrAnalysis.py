import pytesseract
from PIL import Image
from langdetect import detect
import os

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def has_indic_chars(text):
    """
    Check if the text contains Indic characters.
    """
    ranges = [
        (0x0900, 0x097F),  # Devanagari
        (0x0980, 0x09FF),  # Bengali
        (0x0A00, 0x0A7F),  # Gurmukhi
        (0x0C00, 0x0C7F),  # Telugu
        (0x0C80, 0x0CFF),  # Kannada
        (0x0D00, 0x0D7F),  # Malayalam
        (0x0B00, 0x0B7F),  # Oriya
        (0x0A80, 0x0AFF)   # Gujarati
    ]
    return any(any(start <= ord(char) <= end for start, end in ranges) for char in text)

def ocr_analysis(image_path):
    """
    Performs OCR analysis on the image.
    """
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    # Perform OCR
    text = pytesseract.image_to_string(image, lang='eng+hin+tam+tel+kan+mal+ben')

    # Detect language
    detected_lang = detect(text) if text.strip() else 'unknown'

    # Check for Indic scripts
    script_score = 1 if has_indic_chars(text) else 0

    return text, detected_lang, script_score

# Example usage
if __name__ == "__main__":
    image_path = 'test2.jpg'  # Replace with your image path
    ocr_text, language, script_score = ocr_analysis(image_path)

    print("OCR Text:", ocr_text)
    print("Detected Language:", language)
    print("Indic Script Score:", script_score)
