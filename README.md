Hand Gesture Detection using Streamlit

This project is a simple application for detecting hand gestures in video files using TensorFlow, Mediapipe, OpenCV, and Streamlit.
Installation

To run the application, you'll need Python installed on your system. You can install the required dependencies using pip:

bash

pip install -r requirements.txt

Usage

To start the application, run the following command:

bash

streamlit run main.py

This will launch a Streamlit web application where you can upload a video file to detect hand gestures.
Features

    Supports video files in MP4, AVI, and WEBM formats.
    Displays the processed video frames with hand gesture detection overlaid.
    Detects and displays a message when a gesture is detected in the video.

File Structure

    main.py: The main Python script containing the Streamlit app.
    requirements.txt: A text file containing the list of Python dependencies.
    README.md: This file providing information about the project.

Dependencies

    OpenCV
    Mediapipe
    TensorFlow
    Streamlit
