import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class IndianColorDetector:

    def __init__(self):
        # Define Indian color palette with names and RGB values
        self.indian_colors = {
            'Saffron': (255, 153, 51),
            'Deep Saffron': (255, 127, 0),
            'Indian Green': (19, 136, 8),
            'India Green': (0, 128, 0),
            'Navy Blue': (0, 0, 128),
            'Sindoor Red': (255, 0, 0),
            'Kumkum Red': (191, 0, 0),
            'Turmeric Yellow': (255, 255, 0),
            'Marigold': (255, 194, 14),
            'Purple': (128, 0, 128),
            'Mehendi Green': (124, 169, 59),
            'Peacock Blue': (51, 161, 201),
            'Gold': (255, 215, 0)
        }
        
    def preprocess_image(self, image):
        """Preprocess image with multiple techniques for better color detection"""
        image_f = np.float32(image)
        denoised = cv2.bilateralFilter(image_f, 9, 75, 75)
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced

    def color_distance(self, color1, color2):
        """Calculate color distance in LAB color space"""
        color1_lab = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0][0]
        color2_lab = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0][0]
        return np.sqrt(np.sum((color1_lab - color2_lab) ** 2))

    def detect_colors_multiscale(self, image_path, scales=[1.0, 0.75, 0.5], tolerance=25):
        """Detect Indian colors at multiple scales"""
        original_image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        results = []
        for scale in scales:
            width = int(image_rgb.shape[1] * scale)
            height = int(image_rgb.shape[0] * scale)
            scaled_image = cv2.resize(image_rgb, (width, height))
            processed_image = self.preprocess_image(scaled_image)
            scale_results = self.process_image_colors(processed_image, tolerance)
            results.append((scale, scale_results))
            
        return results, image_rgb

    def process_image_colors(self, image, tolerance):
        """Process image to detect Indian colors and their locations"""
        height, width = image.shape[:2]
        detected_colors = {}
        
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        grid_size = 5
        y_coords = np.linspace(0, height-1, num=height//grid_size, dtype=int)
        x_coords = np.linspace(0, width-1, num=width//grid_size, dtype=int)
        
        for y in y_coords:
            for x in x_coords:
                pixel_color = image[y, x]
                
                for color_name, color_value in self.indian_colors.items():
                    if self.color_distance(pixel_color, color_value) < tolerance:
                        detected_colors[color_name] = detected_colors.get(color_name, 0) + 1
        
        return detected_colors

    def analyze_color_distribution(self, detected_colors, image_shape):
        """Analyze the distribution of detected colors"""
        analysis = {}
        total_pixels = image_shape[0] * image_shape[1]
        
        for color_name, count in detected_colors.items():
            percentage = (count * 100) / total_pixels
            confidence = count / total_pixels  # Calculate confidence
            
            analysis[color_name] = {
                'count': count,
                'percentage': percentage,
                'confidence': confidence  # Include confidence in the analysis
            }
        
        return analysis

    def visualize_results(self, image, detected_colors):
        """Visualize detected colors on the image"""
        vis_image = image.copy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(vis_image)
        
        for color_name, stats in detected_colors.items():
            if stats['count'] > 0:
                color_rgb = np.array(self.indian_colors[color_name]) / 255.0
                ax.scatter(np.random.randint(0, image.shape[1], stats['count']), 
                           np.random.randint(0, image.shape[0], stats['count']), 
                           c=[color_rgb], label=color_name, alpha=0.6, s=20)
        
        ax.set_title('Detected Indian Colors')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        
        plt.tight_layout()
        # plt.show()

    def run_analysis(self, image_path):
        """Run complete color analysis"""
        results, original_image = self.detect_colors_multiscale(image_path)
        
        filtered_colors = {}
        for scale, detected_colors in results:
            print(f"\nResults for scale {scale}:")
            analysis = self.analyze_color_distribution(detected_colors, original_image.shape)
            
            for color_name, stats in analysis.items():
                print(f"\n{color_name}:")
                print(f"  Pixels detected: {stats['count']}")
                print(f"  Coverage: {stats['percentage']:.2f}%")
                print(f"  Confidence: {stats['confidence']:.2f}")
                
                # Filter based on confidence and coverage criteria
                if stats['confidence']>=0.035 and stats['count']>=46500:  # stats['percentage'] > 0.015 and   # Updated coverage threshold
                    filtered_colors[color_name] = stats
            
            # Visualize results for the filtered colors only
            self.visualize_results(original_image, filtered_colors)

        return list(filtered_colors.keys())  # Return a unique list of detected colors

# Example usage
def main():
    detector = IndianColorDetector()
    image_path = 'test1.jpg'  # Replace with your image path
    detected_colors = detector.run_analysis(image_path)
    
    print("\nFiltered detected colors:", detected_colors)

if __name__ == "__main__":
    main()
