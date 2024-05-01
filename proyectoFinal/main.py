# Import Libraries
import cv2
import numpy as np
import os 
import matplotlib as plt
import time 
import mediapipe as mp

# Model Detection + Drawing

mp_holistic = mp.solutions.holistic #Holistic Model
mp_drawing = mp.solutions.drawing_utils #Drawing

#Detection Model 
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results 

#Draw Hands (Colors are in BGR format)
def draw_landmarks(image, results):
    # Draw left hand 
    mp_drawing.draw_landmarks(
    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2), #Circles
    mp_drawing.DrawingSpec(color=(127, 0, 255), thickness=2) #Lines
    ) 
    # Draw right hand   
    mp_drawing.draw_landmarks(
    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2), #Circles
    mp_drawing.DrawingSpec(color=(255, 127, 0), thickness=2) #Lines
    ) 


# Draw Hands with bounding box (Colors are in BGR format)
def draw_landmarksB(image, results, padding=40):  # padding parameter with a default of 20 pixels
    # Draw left hand
    if results.left_hand_landmarks:
        # Calculate the bounding box for the left hand with padding
        x_min = min([lm.x for lm in results.left_hand_landmarks.landmark]) * image.shape[1] - padding
        x_max = max([lm.x for lm in results.left_hand_landmarks.landmark]) * image.shape[1] + padding
        y_min = min([lm.y for lm in results.left_hand_landmarks.landmark]) * image.shape[0] - padding
        y_max = max([lm.y for lm in results.left_hand_landmarks.landmark]) * image.shape[0] + padding

        # Ensure coordinates are within image boundaries
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)

        # Draw the rectangle with padding
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Green rectangle
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),  # Circles
            mp_drawing.DrawingSpec(color=(127, 0, 255), thickness=2)  # Lines
        )
    
    # Draw right hand
    if results.right_hand_landmarks:
        # Calculate the bounding box for the right hand with padding
        x_min = min([lm.x for lm in results.right_hand_landmarks.landmark]) * image.shape[1] - padding
        x_max = max([lm.x for lm in results.right_hand_landmarks.landmark]) * image.shape[1] + padding
        y_min = min([lm.y for lm in results.right_hand_landmarks.landmark]) * image.shape[0] - padding
        y_max = max([lm.y for lm in results.right_hand_landmarks.landmark]) * image.shape[0] + padding

        # Ensure coordinates are within image boundaries
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)

        # Draw the rectangle with padding
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # Blue rectangle
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),  # Circles
            mp_drawing.DrawingSpec(color=(255, 127, 0), thickness=2)  # Lines
        )

    
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        #Selfie View 
        #Since we are changing the view, the setting for the left hand will apear on the right hand on screan and viceversa. 
        frame = cv2.flip(frame,1)
        # Make detections
        frame, results = mediapipe_detection(frame, holistic)
        # Extract Keypoints 
        # Get results in one single array, if doesnt exist add 0s. 
        left_hand_nparray = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        right_hand_nparray = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        """

        #Normalizar a 200x200?

        #Analizar comportamiento
        print('Left Hand np array')
        print(len(left_hand_nparray))
        print('-------------------')
        print(left_hand_nparray)
        print('-------------------')

        print('Right Hand np array')
        print(len(right_hand_nparray))
        print('-------------------')
        print(right_hand_nparray)
        print('-------------------')
        """
        # Draw landmarks
        draw_landmarksB(frame, results)  
        # Show to screen
        cv2.putText(frame, "Press 'q' to end", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        # Check if 'q' is pressed to end hand tracking.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

