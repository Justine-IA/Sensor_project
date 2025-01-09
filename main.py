import cv2
import cv2.aruco as aruco
from MyDetectionMethods import MyDetectionMethods, ColorDetector
import numpy as np
import math

def recognition_needle(width, height, contour, box, frame, rect ,centroid, color_needle):

                if width> height:
                    stock = width
                    width = height
                    height = stock
                
                if width > 0.8 and height > 2.3 and width <= 1.5 and height <= 3.5:
                        
                    cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                    #cv2.circle(frame, centroid, radius=2, color=(0, 0, 255), thickness=-1)

                    bottom_left_corner = tuple(box[3])  


                    text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)
                    # angle = round(rect[2], 2) 


                    # Display the angle and classification near the bottom-left corner
                    text = f"Large, {color_needle}" #, Angle:{angle}°
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


                elif width > 0.3 and height > 1.7 and width <0.8 and height<2.4 :
                        
                    cv2.drawContours(frame, [contour], 0, (255, 0, 0), 2)

                    bottom_left_corner = tuple(box[3])  


                    text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)
                    # angle = round(rect[2], 2) 

                    text = f"small, {color_needle}" #, Angle:{angle}°
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


                # elif width>1.5 and height > 2 and width <=3 and height <=3:
                #     if width > 1.9 and height>1.9 and width<2.2 and height<2.2:
                #         bottom_left_corner = tuple(box[3])  
                #         text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)


                #         text = f"Aruco Marker" # , Angle:{angle} Width:{width:.1f}, Height:{height:.1f},
                #         cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        
                    

                #     else:
                #         # cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)

                #         cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

                #         bottom_left_corner = tuple(box[3])  

                #         # Offset the text slightly above the bottom-left corner to avoid overlap
                #         text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)
                #         angle = round(rect[2], 1) 


                #         text = f"Multiple needle superposed, Angle:{angle}° " # , Width:{width:.1f}, Height:{height:.1f},
                #         cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


                # elif width>0.3 and height >0.7 and width<1.3 and height <2 :
                #     # Draw the circle representing the center of the rectangle in the frame
                #     cv2.circle(frame, centroid, radius=2, color=(0, 0, 255), thickness=4)
                #     cv2.drawContours(frame, [box], 0, (255, 0, 255), 2)

                #     cv2.drawContours(frame, [contour], 0, (255, 0, 255), 2)

                #     bottom_left_corner = tuple(box[3])  

                #     # Offset the text slightly above the bottom-left corner to avoid overlap
                #     text_position = (bottom_left_corner[0], bottom_left_corner[1] + 20)
                #     angle = round(rect[2], 1) 


                #     # Write the size near the centroid
                #     text = f"BROKEN NEEDLE, {color_needle}"
                #     cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)




def main():
    color_detector = ColorDetector()

    
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

 
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    aruco_size = None
    pixel_to_cm_ratio = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break


        edges, canny_contours = MyDetectionMethods.canny_filter(frame)
        # cv2.drawContours(frame, canny_contours, -1, (0, 0, 255), 2) 
        head_edges, head_contours = MyDetectionMethods.head_detection(frame)

        cv2.imshow("Canny Filter", edges)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame  to have corners and ids
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        print(corners)

        if ids is not None and len(corners) > 0 and len(corners[0]) > 0:

            top_left_corner = corners[0][0][0]
            top_right_corner = corners[0][0][1]

            print("top left : ", top_left_corner)
            print("top right : ", top_right_corner)

            aruco_size = math.sqrt((top_right_corner[0] - top_left_corner[0])**2 +
                       (top_right_corner[1] - top_left_corner[1])**2)

            print("aruco size", aruco_size)

            pixel_to_cm_ratio = 1.8/aruco_size

            for contour in canny_contours:
                rect = cv2.minAreaRect(contour)

                box = cv2.boxPoints(rect)  

                box = np.int32(box)  


                centroid = (int(rect[0][0]), int(rect[0][1]))

  

                width = round(rect[1][0] * pixel_to_cm_ratio, 2)
                height = round(rect[1][1] * pixel_to_cm_ratio, 2)

                #we create a filter to filter out too small or too big object as we want 
                color = color_detector.process_detections(frame, canny_contours, head_contours, pixel_to_cm_ratio, centroid)

                recognition_needle(width, height, contour, box, frame, rect ,centroid, color)




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