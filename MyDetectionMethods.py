import cv2
import numpy as np


#creating class with the two filters in it to then analyze and filter images 
#to isolate object with the main file 
class MyDetectionMethods:
    @staticmethod
    def canny_filter(image_data, lower_threshold=70, upper_threshold=190):
        # Convert to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)


    
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)


        adaptive_thresh = cv2.adaptiveThreshold(
            edges, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted mean for local thresholding
            cv2.THRESH_BINARY_INV,  # Inverted binary threshold
            9, 2  # Block size and constant to subtract
        )


        # edges = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)



        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours

    @staticmethod
    def head_detection(image_data, lower_threshold=100, upper_threshold=190):
        # Convert to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)


        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted mean for local thresholding
            cv2.THRESH_BINARY_INV,  # Inverted binary threshold
            11, 2  # Block size and constant to subtract
        )

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(adaptive_thresh, (3, 3), 1.4)

        thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Filter using contour area and remove small noise
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 200:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

        # Morph close and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edges = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


        # Apply Canny edge detection
        edges = cv2.Canny(edges, lower_threshold, upper_threshold)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours






class ColorDetector:
    def __init__(self):
        # Définition des plages de couleurs en HSV
        self.color_ranges = {
            'red': [
                {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}
            ],
            'blue': [
                {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
            ],
            'green': [
                {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])}
            ],
            'yellow': [
                {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
            ],
            'black': [
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 30])}
            ],
            'white': [
                {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])}
            ]
        }

    def detect_color(self, roi):
        """
        Détecte la couleur dominante dans une région d'intérêt (ROI)
        """
        # Conversion en HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculer le pourcentage de pixels pour chaque couleur
        color_percentages = {}
        
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            
            # Pour gérer les couleurs avec plusieurs plages (comme le rouge)
            for range_dict in ranges:
                current_mask = cv2.inRange(hsv_roi, range_dict['lower'], range_dict['upper'])
                mask = cv2.bitwise_or(mask, current_mask)
            
            color_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            percentage = (color_pixels / total_pixels) * 100
            color_percentages[color_name] = percentage
        
        # Retourner la couleur avec le plus haut pourcentage si elle dépasse un seuil
        max_color = max(color_percentages.items(), key=lambda x: x[1])
        if max_color[1] > 15:  # Seuil de 15%
            return max_color[0]
        return "color"

         
    def process_detections(self, frame, shape_contours, head_contours, pixel_to_cm_ratio, centroid):#self, frame, contours, pixel_to_cm_ratio):
        """
        Traite les détections en utilisant les contours de forme et de couleur séparément
        """
        self.detections = []
        total_objects = {'Large': 0, 'Small': 0, 'Multiple': 0, 'Broken': 0}
        colors_found = set()

        # Créer un dictionnaire pour stocker les couleurs détectées par position
        color_positions = {}
        
        # D'abord, analyser les contours des têtes pour la couleur
        for contour in head_contours:
            # Obtenir le centre du contour de la tête
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Extraire la ROI autour de ce contour
                x, y, w, h = cv2.boundingRect(contour)
                head_roi = frame[y:y+h, x:x+w]
                
                if head_roi is not None and head_roi.size > 0:
                    color = self.detect_color(head_roi)
                    if color != "indéterminé":
                        # Stocker la couleur avec sa position
                        color_positions[(cx, cy)] = color
                        colors_found.add(color)



            # Trouver la couleur la plus proche de ce contour
            color = "indéterminé"
            min_distance = float('inf')
            for (cx, cy), col in color_positions.items():
                dist = ((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2) ** 0.5
                if dist < min_distance and dist < 50:  # Seuil de distance de 50 pixels
                    min_distance = dist
                    color = col

            

        return color
        
    
    def draw_detections(self, frame):
        """
        Dessine toutes les détections sur l'image
        """
        # Dessiner les contours et les informations pour chaque détection
        for det in self.detections:
            cv2.drawContours(frame, [det['contour']], 0, det['color_contour'], 2)
            
            # Ajouter le texte avec la catégorie et la couleur
            bottom_left_corner = tuple(det['box'][3])
            text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)
            text = f"{det['category']}"
            if det['color'] != "indéterminé":
                text += f" - {det['color']}"
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Ajouter un récapitulatif en haut de l'image
        y_pos = 30
        cv2.putText(frame, "Detected objects:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for det in self.detections:
            y_pos += 20
            summary = f"{det['category']} ({det['color']})"
            cv2.putText(frame, summary, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
