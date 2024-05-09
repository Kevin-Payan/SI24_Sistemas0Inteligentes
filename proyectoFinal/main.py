import cv2
import numpy as np
import os 
import matplotlib as plt
import time 
import mediapipe as mp

from tensorflow.python.keras.models import load_model

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


# Draw Hands with bounding box (Colors are in BGR format)
def draw_landmarks(image, results, padding=50):  # padding parameter 

    hand_coordinates = {}  # Dictionary for bounding box coordinates

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

        # Store the left hand coordinates
        hand_coordinates['left_hand'] = {'x_min': int(x_min), 'y_min': int(y_min), 'x_max': int(x_max), 'y_max': int(y_max)}

        # Draw the rectangle with padding
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)  # Green rectangle
        
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

         # Store the right hand coordinates
        hand_coordinates['right_hand'] = {'x_min': int(x_min), 'y_min': int(y_min), 'x_max': int(x_max), 'y_max': int(y_max)}

        # Draw the rectangle with padding
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # Blue rectangle
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),  # Circles
            mp_drawing.DrawingSpec(color=(255, 127, 0), thickness=2)  # Lines
        )

    return hand_coordinates  # Return a dictionary that points to two other dictionaries containing the coordinates


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        #Selfie View 
        #Since we are changing the view, the setting for the left hand will apear on the right hand on screan and viceversa?
        frame = cv2.flip(frame,1)
        
        frame_for_saving = frame.copy()

        # Make detections
        frame, results = mediapipe_detection(frame, holistic)

        # Get & Draw landmarks
        hand_coordinates = draw_landmarks(frame, results)  

        if frame.size > 0:
            cv2.imwrite('Before.jpg', frame) 

        if 'left_hand' in hand_coordinates:
            left_hand_box = hand_coordinates['left_hand']
            cropped_frame_lh = frame_for_saving[left_hand_box['y_min']:left_hand_box['y_max'], left_hand_box['x_min']:left_hand_box['x_max']]
            # cropped_framelh -> Predict
        else:
            cropped_frame_lh = frame_for_saving # ¿Que hacer/enviar cuando no detecta nada?
            # no predict

        if 'right_hand' in hand_coordinates:
            right_hand_box = hand_coordinates['right_hand']
            cropped_frame_rh = frame_for_saving[right_hand_box['y_min']:right_hand_box['y_max'], right_hand_box['x_min']:right_hand_box['x_max']]
        else:
            cropped_frame_rh = frame_for_saving # ¿Que hacer/enviar cuando no detecta nada?

        # cropped_framelh -> Predict
        # cropped_framerh -> Predict

        # Save the current frame for visualization (borrar esto ya que sirva)
        #Ya que sirva mandar a prediccion y no guardar, guardar lo hace lento. Guardar solo para visualizar
        if frame.size > 0:
            cv2.imwrite('Left_Hand.jpg', cropped_frame_lh)  
            cv2.imwrite('Right_Hand.jpg', cropped_frame_rh) 
        else:
            print("Failed to save.")

        # Show to screen
        cv2.putText(frame, "Press 'q' to end", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        # Check if 'q' is pressed to end hand tracking.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



    """ 
        # Get results in one single array, if doesnt exist add 0s. 
        left_hand_nparray = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        right_hand_nparray = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
       
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

"""
    # Check if the left hand data is available
if 'left_hand' in hand_coordinates:
    left_hand_box = hand_coordinates['left_hand']
    print("Left Hand Bounding Box:")
    print(f"X Min: {left_hand_box['x_min']}")
    print(f"Y Min: {left_hand_box['y_min']}")
    print(f"X Max: {left_hand_box['x_max']}")
    print(f"Y Max: {left_hand_box['y_max']}")
"""

"""
# Save the current frame (hand drawing)
        cv2.imwrite('captured_framee.jpg', frame)  
        print("Frame saved.")
"""