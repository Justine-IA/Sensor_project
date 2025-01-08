import cv2
import numpy as np


#creating class with the two filters in it to then analyze and filter images 
#to isolate object with the main file 
class MyDetectionMethods:
    @staticmethod
    def canny_filter(image_data, lower_threshold=100, upper_threshold=190):
        # Convert to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)





        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)  # Enhanced grayscale image

        # Step 2: Optionally, apply Adaptive Thresholding for shadow reduction
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted mean for local thresholding
            cv2.THRESH_BINARY_INV,  # Inverted binary threshold
            11, 2  # Block size and constant to subtract
        )

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(adaptive_thresh, (3, 3), 1.4)

        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Filter using contour area and remove small noise
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

        # Morph close and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edges = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        



        # Apply Canny edge detection
        # edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)



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

        return contours


