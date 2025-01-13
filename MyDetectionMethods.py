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
        # HSV color range definitions
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
        Detect the most dominant color out of the ROI  
        """
        # Convertion from BGR to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculating the pourcentage of pixel for each range of color using a mask 
        color_percentages = {}
        
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            
            # Range has 2 ranges because it's a circular spectrum, it begins and ends with red, so we need to apply the mask for each range
            for range_dict in ranges:
                current_mask = cv2.inRange(hsv_roi, range_dict['lower'], range_dict['upper'])
                mask = cv2.bitwise_or(mask, current_mask)
            
            color_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            percentage = (color_pixels / total_pixels) * 100
            color_percentages[color_name] = percentage
        
        # Return the color with the highest pourcentage if it's above the threshold
        max_color = max(color_percentages.items(), key=lambda x: x[1])
        if max_color[1] > 15:  # Seuil de 15%
            return max_color[0]
        return "color"

         
    def process_detections(self, frame, head_contours, centroid):#self, frame, contours, pixel_to_cm_ratio):
        self.detections = []
        total_objects = {'Large': 0, 'Small': 0, 'Multiple': 0, 'Broken': 0}
        colors_found = set()

        color_positions = {}

        for contour in head_contours:
            # cv2.moments() return the center of the contours
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Extract the ROI aounrd the contour
                x, y, w, h = cv2.boundingRect(contour)
                head_roi = frame[y:y+h, x:x+w]
                
                if head_roi is not None and head_roi.size > 0:
                    color = self.detect_color(head_roi)
                    if color != "indetermined":
                        # Storing the color and its position
                        color_positions[(cx, cy)] = color
                        colors_found.add(color)



            # Find the nearest color to the centroid from the positions returned calculating the euclidian distance
            color = "indetermined"
            min_distance = float('inf')
            for (cx, cy), col in color_positions.items():
                dist = ((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2) ** 0.5
                if dist < min_distance and dist < 50:  # the distance threshold is 50 pixels
                    min_distance = dist
                    color = col

        return color
