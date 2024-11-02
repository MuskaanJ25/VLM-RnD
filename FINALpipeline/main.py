import spacy
from PIPEcolor import IndianColorDetector
# from PIPEgeo import detect_landmarks
from PIPEobject_detection import detect_objects
from PIPEocrAnalysis import ocr_analysis
from PIPEsymbol_detection import detect_symbols

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Function to perform NLP on caption and check for similarities
def check_similarity(detected_items, caption):
    doc = nlp(caption)
    for item in detected_items:
        if item.lower() in doc.text.lower():
            return True
    return False

# Main function to calculate the score
def main(image_path, caption):
    # Initialize scores for each component
    scores = {
        'landmarks': 0,
        'objects': 0,
        'symbols': 0,
        'color': 0,
        'ocr': 0
    }

    # Detect landmarks
    # landmarks = detect_landmarks(image_path)
    # print(f"Landmarks detected: {len(landmarks)}")
    # if len(landmarks) > 0 and check_similarity(landmarks, caption):
    #     scores['landmarks'] = 1

    # Detect objects
    objects = detect_objects(image_path)
    print(f"Objects detected: {len(objects)}")
    if len(objects) > 0 and check_similarity(objects, caption):
        scores['objects'] = 1

    # Detect symbols
    symbols,result_info = detect_symbols(image_path,"./symbols")
    print(f"Symbols detected: {len(symbols)}")
    if len(symbols) > 0 and check_similarity(symbols, caption):
        scores['symbols'] = 1

    # Perform color analysis
    detector = IndianColorDetector()
    image_path = 'test1.jpg'  # Replace with your image path
    detected_colors = detector.run_analysis(image_path)
    color_detected = detected_colors
    print(f"Color detected: {color_detected}")
    if color_detected and check_similarity(color_detected, caption):
        scores['color'] = 1

    # Perform OCR analysis
    ocr_result, language, script_score = ocr_analysis(image_path)
    print(f"OCR result: {ocr_result}")
    if ocr_result and check_similarity([ocr_result], caption):
        scores['ocr'] = 1

    # Weights for each component
    weights = {
        'objects': 0.30,
        'symbols': 0.4,
        'color': 0.20,
        'ocr': 0.1
    }

    # Calculate final weighted score
    final_score = sum(scores[component] * weights[component] for component in scores) * 10

    print(f"Final Score: {final_score}")
    return final_score


if __name__ == "__main__":
    image_path = "test1.jpg"
    caption = "Amarnath cave temple in mountain Indian region."
    main(image_path, caption)
