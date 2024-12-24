# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# import os

# model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'random_forest_model.pkl')
# # Load the model
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# cap = cv2.VideoCapture(1)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
#     9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
#     17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y',
#     24: 'Hello', 25: 'my', 26: 'is', 27: 'name', 28: 'nice', 29: 'meet',
#     30: 'you', 31: 'what', 32: 'your'
# }
# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)


# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'random_forest_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(1)  # Use external camera (index 1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y',
    24: 'Hello', 25: 'my', 26: 'is', 27: 'name', 28: 'nice', 29: 'meet',
    30: 'you', 31: 'what', 32: 'your'
}

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        continue

    # Prepare frame
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    # Store feature data
    data_aux = []
    x_ = []
    y_ = []

    # Process detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract x, y coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Calculate relative positions for features
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x-coordinates
                data_aux.append(y - min(y_))  # Normalize y-coordinates

        # Pad the feature vector to match training data (84 features for 2 hands)
        while len(data_aux) < 84:
            data_aux.append(0.0)  # Add zeros if features are missing
            

        
        # Ensure shape consistency with training
        data_aux = np.asarray(data_aux).reshape(1, -1)  # Reshape for prediction

        # Debugging: Print features and model output
        print("Feature Shape:", data_aux.shape)
        print("Feature Values:", data_aux)  # Features extracted for prediction
        print("Model Prediction Probabilities:", model.predict_proba(data_aux))  # Probabilities for each class

        # Predict using the trained model
        prediction = model.predict(data_aux)
        print("Raw Prediction:", prediction)  # Print raw predicted label

        predicted_character = labels_dict[int(prediction[0])]



        # Draw prediction on the frame
        cv2.putText(frame, predicted_character, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()