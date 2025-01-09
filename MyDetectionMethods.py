import cv2
import numpy as np

# Ajout dans MyDetectionMethods.py

class ColorDetector:
    def __init__(self):
        # Définition des plages de couleurs en HSV
        self.color_ranges = {
            'rouge': [
                {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}
            ],
            'bleu': [
                {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
            ],
            'vert': [
                {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])}
            ],
            'jaune': [
                {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
            ],
            'noir': [
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 30])}
            ],
            'blanc': [
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
        return "indéterminé"

    def extract_head_roi(self, contour, frame):
        """
        Extracts a more precise ROI for the pushpin/needle head
        """

        """
    Extrait une ROI simplifiée basée sur les points de la boîte englobante.
    """
        # Obtenir la boîte englobante
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        
        # # Trier les points par coordonnée y
        # box_sorted = sorted(box, key=lambda p: p[1])
        
        # # Prendre les deux points supérieurs
        # top_points = box_sorted[:2]
        
        # # Calculer les dimensions de la ROI
        # x_min = max(0, int(min(p[0] for p in top_points) - 5))
        # x_max = min(frame.shape[1], int(max(p[0] for p in top_points) + 5))
        # y_min = max(0, int(min(p[1] for p in top_points)))
        # y_max = min(frame.shape[0], int(y_min + (box_sorted[3][1] - box_sorted[0][1]) / 4))
        
        # # Vérifier que la ROI est valide
        # if x_min >= x_max or y_min >= y_max:
        #     return None
            
        # # Extraire la ROI
        # roi = frame[y_min:y_max, x_min:x_max]
        
        # # Vérifier que la ROI n'est pas vide
        # if roi.size == 0:
        #     return None
            
        # # Dessiner le rectangle de la ROI pour le débogage
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        
        # return roi

        # # Get rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #x1, y1, x2, y2 = cv2.boundingRect(contour)
        #roi = frame[y1:y1+y2, x1:x1+x2]
        
        # Sort points by y-coordinate to identify top points
        box_sorted = sorted(box, key=lambda p: p[1])
        top_points = box_sorted[:2]
        
        # Calculate center and angle of the object
        center = rect[0]
        angle = rect[2]
        
        # Determine if object is vertical or horizontal
        is_vertical = abs(angle) < 45 or abs(angle) > 135
        
        # Calculate ROI dimensions
        if is_vertical:
            # For vertical objects, take top 20% of height
            roi_height = int(rect[1][1] * 0.2)  # Reduced from 1/4 to 1/5
            roi_width = int(rect[1][0] * 1.2)   # Add 20% margin to width
        else:
            # For horizontal objects, take leftmost/rightmost 20%
            roi_height = int(rect[1][0] * 0.2)
            roi_width = int(rect[1][1] * 1.2)
        
        # Calculate ROI position
        x_center = int(center[0])
        y_center = int(min(p[1] for p in top_points))  # Use highest point
        
        # Define ROI boundaries with proper boundary checking
        x1 = max(0, x_center - roi_width//2)
        y1 = max(0, y_center)
        x2 = min(frame.shape[1], x_center + roi_width//2)
        y2 = min(frame.shape[0], y_center + roi_height)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Verify ROI is valid
        if roi.size == 0:
            return None
            
        # Draw ROI rectangle for debugging (using a thinner line)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        return roi


        # """
        # Extrait la ROI de la tête d'une aiguille/punaise en utilisant boundingRect
        # """
        # # Obtenir le rectangle englobant
        # x, y, w, h = cv2.boundingRect(contour)
        
        # # Calculer la hauteur de la ROI (1/4 de la hauteur totale)
        # roi_height = h // 4
        
        # # Définir les limites de la ROI en se concentrant sur le haut de l'objet
        # roi_y = max(0, y)  # Point de départ en y
        # roi_h = min(roi_height, frame.shape[0] - roi_y)  # Hauteur de la ROI
        
        # # Ajouter une petite marge horizontale (10% de la largeur)
        # margin = int(w * 0.1)
        # roi_x = max(0, x - margin)
        # roi_w = min(w + 2*margin, frame.shape[1] - roi_x)
        
        # # Extraire la ROI
        # roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # # Vérifier que la ROI est valide
        # if roi.size == 0:
        #     return None
        
        # # Pour le débogage, dessiner le rectangle de la ROI sur l'image
        # cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 1)
        
        # return roi



        # """
        # Extrait la ROI de la tête d'une aiguille/punaise
        # """
        # # Obtenir la boîte englobante
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        
        # # Trier les points par coordonnée y (vertical)
        # box = sorted(box, key=lambda point: point[1])
        
        # # Les deux points supérieurs de la box (tête de l'aiguille)
        # top_points = box[:2]
        
        # # Calculer les dimensions de la ROI
        # x_coords = [p[0] for p in top_points]
        # y_coords = [p[1] for p in top_points]
        
        # # Définir les limites de la ROI
        # min_x = max(0, int(min(x_coords) - 5))
        # max_x = min(frame.shape[1], int(max(x_coords) + 5))
        # min_y = max(0, int(min(y_coords)))
        # height = int((box[3][1] - box[0][1]) / 4)
        # max_y = min(frame.shape[0], min_y + height)
        
        # if min_x >= max_x or min_y >= max_y:
        #     return None
            
        # roi = frame[min_y:max_y, min_x:max_x]
        
        # if roi.size == 0:
        #     return None
            
        # # Pour le débogage, dessiner la ROI sur l'image
        # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
            
        # return roi
    
    def process_detections(self, frame, shape_contours, head_contours, pixel_to_cm_ratio):#self, frame, contours, pixel_to_cm_ratio):
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

        # Ensuite, analyser les contours de forme
        for contour in shape_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            centroid = (int(rect[0][0]), int(rect[0][1]))
            
            width = round(rect[1][0] * pixel_to_cm_ratio, 2)
            height = round(rect[1][1] * pixel_to_cm_ratio, 2)

            if width > height:
                width, height = height, width

            # Déterminer la catégorie de l'objet
            if width > 0.8 and height > 2.5 and width <= 1.5 and height <= 3.5:
                category = "Large"
                color_contour = (0, 0, 255)
            elif width > 0.3 and height > 2 and width < 0.8 and height < 2.4:
                category = "Small"
                color_contour = (255, 0, 0)
            elif width > 1.5 and height > 2 and width <= 4 and height <= 4:
                category = "Multiple"
                color_contour = (0, 255, 255)
            elif width > 0.2 and height > 0.4 and width < 1.3 and height < 2.4:
                category = "Broken"
                color_contour = (255, 0, 255)
            else:
                continue

            # Trouver la couleur la plus proche de ce contour
            color = "indéterminé"
            min_distance = float('inf')
            for (cx, cy), col in color_positions.items():
                dist = ((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2) ** 0.5
                if dist < min_distance and dist < 50:  # Seuil de distance de 50 pixels
                    min_distance = dist
                    color = col

            total_objects[category] += 1
            
            # Stocker la détection
            self.detections.append({
                'category': category,
                'color': color,
                'position': centroid,
                'box': box,
                'contour': contour,
                'color_contour': color_contour,
                'width': width,
                'height': height
            })

        return total_objects, colors_found
        
        # """
        # Traite toutes les détections dans l'image
        # """
        # self.detections = []
        # total_objects = {'Large': 0, 'Small': 0, 'Multiple': 0, 'Broken': 0}
        # colors_found = set()

        # for contour in contours:
        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     centroid = (int(rect[0][0]), int(rect[0][1]))
            
        #     width = round(rect[1][0] * pixel_to_cm_ratio, 2)
        #     height = round(rect[1][1] * pixel_to_cm_ratio, 2)

        #     if width > height:
        #         width, height = height, width

        #     # Détecter la couleur
        #     color = "indéterminé"
        #     if width > 0.3:  # Seulement pour les objets assez grands
        #         head_roi = self.extract_head_roi(contour, frame)
        #         if head_roi is not None:
        #             color = self.detect_color(head_roi)
        #             if color != "indéterminé":
        #                 colors_found.add(color)

        #     # Classifier l'objet
        #     if width > 0.8 and height > 2.5 and width <= 1.5 and height <= 3.5:
        #         category = "Large"
        #         color_contour = (0, 0, 255)
        #     elif width > 0.3 and height > 2 and width < 0.8 and height < 2.4:
        #         category = "Small"
        #         color_contour = (255, 0, 0)
        #     elif width > 1.5 and height > 2 and width <= 4 and height <= 4:
        #         category = "Multiple"
        #         color_contour = (0, 255, 255)
        #     elif width > 0.2 and height > 0.4 and width < 1.3 and height < 2.4:
        #         category = "Broken"
        #         color_contour = (255, 0, 255)
        #     else:
        #         continue

        #     total_objects[category] += 1
            
        #     # Stocker la détection
        #     self.detections.append({
        #         'category': category,
        #         'color': color,
        #         'position': centroid,
        #         'box': box,
        #         'contour': contour,
        #         'color_contour': color_contour,
        #         'width': width,
        #         'height': height
        #     })

        # return total_objects, colors_found
    
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


        edges = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours
    
    @staticmethod
    def tip_needle(image_data, lower_threshold=100, upper_threshold=190):
       # Convert to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)


    
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)


        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours

    @staticmethod
    def head_detection_2(image_data):
        """
        Détecte les contours des têtes en utilisant des plages HSV pour les couleurs spécifiées.
        """
        hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)

        # Plages de couleurs HSV
        color_ranges = {
            "Rouge": [(0, 50, 50), (10, 255, 255)],   # Rouge clair
            "Rouge (foncé)": [(170, 50, 50), (180, 255, 255)],  # Rouge foncé
            "Bleu": [(100, 150, 50), (130, 255, 255)],
            "Jaune": [(20, 100, 100), (30, 255, 255)],
            "Vert": [(40, 50, 50), (80, 255, 255)],
            "Noir": [(0, 0, 0), (180, 255, 50)],
            "Blanc": [(0, 0, 200), (180, 50, 255)],
        }

        masks = {}
        contours_by_color = {}

        for color, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Nettoyage du masque
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Extraction des contours
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            masks[color] = cleaned_mask
            contours_by_color[color] = contours

        return masks, contours_by_color


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

    @staticmethod
    def binarization(image_data, threshold_value=127):
        # Convert to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours





    #   # Convert to grayscale
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    #     gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)


    #     adaptive_thresh = cv2.adaptiveThreshold(
    #         gray, 255, 
    #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted mean for local thresholding
    #         cv2.THRESH_BINARY_INV,  # Inverted binary threshold
    #         11, 2  # Block size and constant to subtract
    #     )

    #     # Apply Gaussian Blur to reduce noise
    #     blurred = cv2.GaussianBlur(adaptive_thresh, (3, 3), 1.4)

    #     thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #     # Filter using contour area and remove small noise
    #     cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #     for c in cnts:
    #         area = cv2.contourArea(c)
    #         if area < 100:
    #             cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    #     # Morph close and invert image
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #     edges = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


    #     # Apply Canny edge detection
    #     # edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    #     # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #     # Find contours
    #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)