import cv2
import os
import numpy as np

def detect_symbols(sample_image_path, symbols_dir, min_scale=0.1, max_scale=1.0, scale_steps=20, threshold=0.5):
    # Load the sample image
    sample_image = cv2.imread(sample_image_path)
    if sample_image is None:
        print(f"Error: Could not load image at {sample_image_path}")
        return
    
    # Convert sample image to grayscale
    gray_sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    sample_height, sample_width = gray_sample_image.shape
    
    # Create a copy for drawing results
    result_image = sample_image.copy()
    
    # Store all detections for non-maximum suppression
    all_detections = []
    
    # Generate scale factors
    scales = np.linspace(min_scale, max_scale, scale_steps)
    
    # Loop through all symbol files
    for symbol_filename in os.listdir(symbols_dir):
        if not symbol_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
            
        symbol_path = os.path.join(symbols_dir, symbol_filename)
        symbol = cv2.imread(symbol_path, 0)  # Load symbol in grayscale
        
        if symbol is None:
            print(f"Error: Could not load symbol at {symbol_path}")
            continue
            
        print(f"Processing symbol: {symbol_filename}")
        
        # Calculate initial symbol size
        symbol_height, symbol_width = symbol.shape
        
        # Multi-scale template matching
        for scale in scales:
            # Calculate new dimensions
            width = int(symbol_width * scale)
            height = int(symbol_height * scale)
            
            # Skip if dimensions are invalid
            if width < 10 or height < 10:  # Too small
                continue
            
            try:
                # Resize the symbol
                resized_symbol = cv2.resize(symbol, (width, height))
                
                # Perform template matching
                result = cv2.matchTemplate(gray_sample_image, resized_symbol, cv2.TM_CCOEFF_NORMED)
                
                # Get positions where result exceeds threshold
                locations = np.where(result >= threshold)
                for pt in zip(*locations[::-1]):  # Switch columns and rows
                    all_detections.append({
                        'pos': pt,
                        'size': (width, height),
                        'conf': result[pt[1], pt[0]],
                        'symbol_name': symbol_filename
                    })
            except cv2.error as e:
                print(f"Error processing scale {scale} for {symbol_filename}: {str(e)}")
                continue
    
    # Non-maximum suppression
    def calculate_iou(box1, box2):
        b1 = [box1['pos'][0], box1['pos'][1], 
              box1['pos'][0] + box1['size'][0], box1['pos'][1] + box1['size'][1]]
        b2 = [box2['pos'][0], box2['pos'][1], 
              box2['pos'][0] + box2['size'][0], box2['pos'][1] + box2['size'][1]]
        
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    # Sort detections by confidence
    all_detections.sort(key=lambda x: x['conf'], reverse=True)
    final_detections = []
    
    # Apply non-maximum suppression
    while len(all_detections) > 0:
        best_detection = all_detections.pop(0)
        final_detections.append(best_detection)
        
        # Filter out overlapping detections
        all_detections = [
            det for det in all_detections
            if calculate_iou(best_detection, det) < 0.3
        ]
    
    # Draw final detections
    detected_symbols = []  # List to store detected symbol names
    for det in final_detections:
        pt = det['pos']
        width, height = det['size']
        cv2.rectangle(result_image, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)
        
        # Display coordinates and symbol name
        text = f"{det['symbol_name']} ({pt[0]}, {pt[1]})"
        cv2.putText(result_image, text, (pt[0], pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Found {det['symbol_name']} at position {pt} with confidence {det['conf']:.2f}")
        detected_symbols.append(det['symbol_name'])  # Add detected symbol name to the list

    # Remove extensions and create a unique list
    detected_symbols_unique = list(set([symbol.split('.')[0] for symbol in detected_symbols]))

    return detected_symbols_unique, result_image  # Return the list of detected symbols and the result image

# Usage example
if __name__ == "__main__":
    # Path configurations
    sample_image_path = 'test4.jpg'  # Change this to your image path
    symbols_dir = './symbols/'        # Change this to your symbols directory
    
    # Print image and symbol information
    sample_image = cv2.imread(sample_image_path)
    if sample_image is not None:
        print(f"Sample image size: {sample_image.shape}")
    
    # Parameters for detection
    params = {
        'min_scale': 0.1,    # Minimum scale to try
        'max_scale': 1.0,    # Maximum scale to try
        'scale_steps': 10,   # Number of different scales to try , 200
        'threshold': 0.6     # Matching threshold (0.0 to 1.0)
    }
    
    # Perform detection
    detected_symbols, result_image = detect_symbols(sample_image_path, symbols_dir, **params)
    
    print("Detected Symbols:", detected_symbols)
    print("Type of result image:", type(result_image))

    # Display results
    # if result_image is not None:
    #     cv2.imshow('Detected Symbols', result_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
        # Optionally save the result
        # cv2.imwrite('result.jpg', result_image)
