import cv2
import numpy as np
import mediapipe as mp
#import tensorflow as tf 
import streamlit as st
from tensorflow.keras.models import load_model

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model

model = load_model('model')

# Function to process each frame
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y, c = frame.shape

    # Get hand landmark prediction
    result = hands.process(frame_rgb)

    # Post-process the result
    gesture_detected = False
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            # Check if any gesture is detected
            if classID != 0:
                gesture_detected = True
                break

    return frame, gesture_detected

# Streamlit app function
def main():
    st.title("Hand Gesture Detection")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "webm"])

    if uploaded_file is not None:
        # Open video file
        cap = cv2.VideoCapture(uploaded_file)

        # Check if the video capture is successful
        if not cap.isOpened():
            st.error("Error: Unable to open the file.")
            return

        # Process the video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, gesture_detected = process_frame(frame)
            if gesture_detected:
                st.write("Gesture Detected!")
            st.image(processed_frame, channels="BGR", use_column_width=True)

        # Release video capture
        cap.release()

if __name__ == "__main__":
    main()
