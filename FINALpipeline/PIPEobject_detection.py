import torch
import cv2
import numpy as np

def detect_objects(image_path):
    # Load pre-trained YOLOv5 model (fine-tune for Indian objects if available)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model

    # Load the image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Extract information (coordinates, labels, confidence scores)
    detections = results.xyxy[0].numpy()  # Format: [x_min, y_min, x_max, y_max, confidence, class]

    # Create a list to store the detection information
    detection_list = []

    # Loop through detections and store relevant info
    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        detection_info = {
            'object': model.names[int(class_id)],
            'coordinates': [x_min, y_min, x_max, y_max],
            'confidence': confidence
        }
        detection_list.append(detection_info)

    object_list = [obj['object'] for obj in detection_list]

    # Return the list of detections
    return object_list

# Example usage:
# detected_objects = detect_objects('test5.jpg')
# print(detected_objects)
# print(type(detect_objects))