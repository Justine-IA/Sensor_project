import cv2
import cv2.aruco as aruco
from MyDetectionMethods import MyDetectionMethods
import numpy as np
def main():
    
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Define ArUco dictionary and parameters
    #this lines define the type of aruco that we are using and want to detect
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    #configure algorithm  to detect aruco markers in images
    parameters = aruco.DetectorParameters()
    #create the detector using the parameters and the dictionnary we defined before
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    #define some variable that we will use later 
    aruco_size = None
    pixel_to_cm_ratio = None
    while True:
        #read the frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # we are using the canny contours function defined in mydetectionmethods.py files
        # to find the contours of object in the frame 
        edges, canny_contours = MyDetectionMethods.canny_filter(frame)
        cv2.drawContours(frame, canny_contours, -1, (0, 0, 255), 2) #if we want to write the contours on the frame

        cv2.imshow("Canny Filter", edges)

        # Detect ArUco markers in the frame using the detector we build earlier to have corners and ids
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Draw bounding boxes around detected markers if ids is present (the ID of the aruco )
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # we print the corners of the ArUco that we got by using the detector
        print(corners)
        #we check to see if we got the corners because we will process them
        if ids is not None and len(corners) > 0 and len(corners[0]) > 0:
            #we take the coordinate of the two top corners
            top_left_corner = corners[0][0][0]
            top_right_corner = corners[0][0][1]

            print("top left : ", top_left_corner)
            print("top right : ", top_right_corner)
            # We take the size in px of the distance between the two top corners
            aruco_size = abs(top_left_corner[0] - top_right_corner[0] )
            print("aruco size", aruco_size)
            #we calcul the ratio pixel to centimeter because 
            # we know that between the two corners there is 2cm 
            pixel_to_cm_ratio = 2/aruco_size

            #we check if we have contours with the canny detector used
            for contour in canny_contours:
                # Get the rotated rectangle for the contour of the object
                #we use the function minAreaRect which find automatically the minimum area
                #that can enclose a contour
                rect = cv2.minAreaRect(contour)
                #we convert rect to the four corner points of the rectangle under the box variable to then draw it later on the frame
                box = cv2.boxPoints(rect)  
                #we convert float into integers 
                box = np.int32(box)  

                #we find the centroid and width with rect[0] having its coordinate x and y
                centroid = (int(rect[0][0]), int(rect[0][1]))
                #same with with and height but it is in rect[1] 
                #and we use pixel to cm ratio to have it in cm and round it
                width = round(rect[1][0] * pixel_to_cm_ratio, 1)
                height = round(rect[1][1] * pixel_to_cm_ratio, 1)

                #we create a filter to filter out too small or too big object as we want 
                if width >= 0.3 and height >= 5 and width<=30 and height <=30:
                    #we draw the circle representig the center of the rectangle in the frame
                    cv2.circle(frame, centroid, radius=2, color=(0, 0, 255), thickness=-1)
                    #same but with the rectangle of the object
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

                    #finally we write on the frame near the centroids the width and height of the object
                    text = f"Width:{width:.1f}, Height:{height:.1f}"
                    cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    


        # Show the frame
        cv2.imshow("ArUco Marker Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
