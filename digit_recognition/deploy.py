import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import joblib

# load the model and scaler
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "mnist_random_forest.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

# streamlit ui
st.title("Digit classifier")
st.write("upload a hand=written digit(0-9)!")


# upload file option
uploaded_file=st.file_uploader("choose an image",type=['jpg','png','jpeg'])

# function to preprocess the uploaded image

def process_uploaded_image(image):
    # Convert to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors
    img = cv2.bitwise_not(img)
    # Show the processed image
    st.image(img, caption="Processed Image", width=150)
    # Flatten and scale the image
    img = img.reshape(1, -1)
    img = scaler.transform(img)
    return img


# Prediction Logic
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    # Preprocess the Image
    processed_image = process_uploaded_image(image)
    # Make prediction
    prediction = model.predict(processed_image)[0]
    # Display prediction
    st.subheader(f"Predicted Digit: {prediction}")