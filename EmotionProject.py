import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import os

# Suppress TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Models
def load_all_models():
    try:
        face_net = cv2.dnn.readNet("Notebooks/opencv_face_detector_uint8.pb", "Notebooks/opencv_face_detector.pbtxt")
        age_net = cv2.dnn.readNet("Notebooks/age_net.caffemodel", "Notebooks/age_deploy.prototxt")
        gender_net = cv2.dnn.readNet("Notebooks/gender_net.caffemodel", "Notebooks/gender_deploy.prototxt")
        emotion_model = load_model('Final_Emotion_Detection_Model.keras')
        return face_net, age_net, gender_net, emotion_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Initialize Models
face_net, age_net, gender_net, emotion_model = load_all_models()

# Labels for age, gender, and emotions
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-80)']
gender_labels = ['Male', 'Female']
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to detect faces within a frame using the face detection model.
def faceBox(face_net, frame, box_scale=60):
    # Define functions for predictions
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detection = face_net.forward()# Runs detection.
    
    h, w = frame.shape[:2] # Gets frame dimensions
    boxes = []# Stores detected face coordinates.
    
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.3:  # Lowered threshold for better detection
            x1 = int(detection[0, 0, i, 3] * w)
            y1 = int(detection[0, 0, i, 4] * h)
            x2 = int(detection[0, 0, i, 5] * w)
            y2 = int(detection[0, 0, i, 6] * h)
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, x1 - box_scale // 2)
            y1 = max(0, y1 - box_scale // 2)
            x2 = min(w, x2 + box_scale // 2)
            y2 = min(h, y2 + box_scale // 2)
            boxes.append((x1, y1, x2, y2))
    return boxes

def predict_age_gender(age_net, gender_net, face):
    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    age_net.setInput(face_blob)
    age_preds = age_net.forward()
    age = age_preds[0].argmax()
    age_label = age_labels[age]
    gender_net.setInput(face_blob)
    gender_preds = gender_net.forward()
    gender = gender_preds[0].argmax()
    gender_label = gender_labels[gender]
    return age_label, gender_label

def predict_emotion(emotion_model, face):
    face_blob = cv2.resize(face, (128, 128))
    if emotion_model.input_shape[-1] == 3 and face_blob.shape[-1] != 3:
        face_blob = cv2.cvtColor(face_blob, cv2.COLOR_GRAY2RGB)
    elif emotion_model.input_shape[-1] == 1 and face_blob.shape[-1] == 3:
        face_blob = cv2.cvtColor(face_blob, cv2.COLOR_RGB2GRAY)
    face_blob = np.expand_dims(face_blob, axis=0)
    face_blob = face_blob / 255.0
    emotion_preds = emotion_model.predict(face_blob)
    emotion = emotion_preds.argmax()
    emotion_label = emotions[emotion]
    return emotion_label

# Streamlit UI
st.markdown(
    f"""
    <style>
    .stApp {{
        color: #FF4545; /* Dark text color for better contrast */
    }}
    .button {{
        background-color: #FF9C73; /* Blue background for buttons */
        color: white; /* White text color */
        border-radius: 5px; /* Rounded corners */
        padding: 10px; /* Padding inside buttons */
    }}
    .button:hover {{
        background-color: #FBD288; /* Darker blue on hover */
    }}
    .header {{
        font-size: 2em; /* Larger title */
        margin-bottom: 20px; /* Space below title */
    }}
    .footer {{
        font-size: 0.8em; /* Smaller footer text */
        text-align: center; /* Center footer */
        margin-top: 20px; /* Space above footer */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='header'>Real-Time Age, Gender, and Emotion Detection</div>", unsafe_allow_html=True)

# Start and Stop Buttons
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("🎥  Start Camera", key="start_camera", help="Start detecting age, gender, and emotion.")
with col2:
    stop_camera = st.button("🛑 Stop Camera", key="stop_camera", help="Stop the camera.")

# Start Webcam if "Start Camera" button is pressed
if start_camera:
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        st.error("Could not open webcam. Check if another application is using it.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = video.read()
            if not ret:
                st.error("Failed to capture frame. Check camera.")
                break

            boxes = faceBox(face_net, frame)
            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    age_label, gender_label = predict_age_gender(age_net, gender_net, face)
                    emotion_label = predict_emotion(emotion_model, face)

                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle

                    # Display labels
                    cv2.putText(frame, f'Age: {age_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Gender: {gender_label}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Emotion: {emotion_label}', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Check if the "Stop Camera" button has been pressed
            if stop_camera:
                break

        video.release()

