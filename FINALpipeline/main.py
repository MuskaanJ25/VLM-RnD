import spacy
from PIPEcolor import IndianColorDetector
# from PIPEgeo import detect_landmarks
from PIPEobject_detection import detect_objects
from PIPEocrAnalysis import ocr_analysis
from PIPEsymbol_detection import detect_symbols
import sys


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
        # 'landmarks': 0,
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
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        caption = " ".join(sys.argv[2:])  # Combine all remaining arguments as the caption
        print("hi")
    else:
        # Default values if no arguments are provided
        image_path = "./Me/Sabarimala_Ayyappa_Temple/Image_67.jpg"
        caption = (
            "This image appears to depict a Hindu temple, showcasing a rich tapestry of architectural style, "
            "religious significance, cultural symbolism, and historical relevance. \n\n"
            "**Architectural Style:** The temple likely adheres to a Dravidian architectural style, "
            "recognizable by its characteristic features:\n\n"
            "* **Gopuram:** The towering gateway, adorned with intricate carvings and sculptures, dominates the structure. \n"
            "* **Pyramidal Shape:** The gopuram, and potentially other temple components, are built with a distinctive pyramidal shape.\n"
            "* **Mandapas:**  Multiple pillared halls, or mandapas, can be observed. These halls serve as spaces for worship, rituals, and community gatherings.\n\n"
            "**Religious Significance:** \n\n"
            "* **Deities:** While the image is a bit blurred, the presence of deities within the temple is implied. The intricate carvings on the gopuram likely depict scenes from Hindu mythology or stories related to the presiding deities. \n"
            "* **Shrine:** The main structure, often called the garbhagriha, houses the primary deity, symbolizing the sacred space where the divine presence is believed to reside.\n\n"
            "**Cultural Symbolism:**\n\n"
            "* **Intricate Carvings:** The temple's exterior and interior are likely adorned with intricate carvings. These carvings often depict mythical creatures, celestial beings, flora and fauna, and scenes from Hindu epics, embodying the rich cultural heritage and artistic traditions of the region.\n"
            "* **Pillar Designs:** The pillars supporting the mandapas are likely intricately carved with detailed patterns. These designs can be symbolic representations of deities, natural elements, or philosophical concepts.\n\n"
            "**Historical Relevance:**\n\n"
            "* **Ancient Tradition:**  The temple's architecture and decoration are indicative of a long and rich history of temple building in India. This particular temple likely represents a significant period of religious and cultural development in the region.\n"
            "* **Community Hub:** Hindu temples are not just places of worship but also community centers that played a central role in social life and cultural expression.\n\n"
            "The temple's overall design and features reflect a profound connection to Indic heritage and spiritual traditions. "
            "The presence of the gopuram, mandapas, intricate carvings, and likely deities all speak to the deep reverence for the divine and the complex cultural and philosophical framework of Hinduism."
        )
    main(image_path, caption)