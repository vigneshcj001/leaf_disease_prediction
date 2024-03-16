import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle

# Load the pre-trained model
model_path = "model.h5"  # Adjust the path based on your model file
model = load_model(model_path)

# Map predicted labels to class names
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Load the scaler (if used during training)
with open('min_max_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Function to preprocess and predict plant disease
def predict_disease(image_path):
    # Preprocess the image for prediction
    image = Image.open(image_path)
    image = image.resize((225, 225))
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, 225, 225, 3))
    image_array = image_array.astype('float32') / 255.0

    # If scaling was used during training, apply it here
    if loaded_scaler:
        image_array = loaded_scaler.transform(image_array)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# Streamlit app
st.title("Plant Disease Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make prediction on the uploaded image
    predicted_label = predict_disease(uploaded_file)

    # Display prediction result
    st.write(f"Prediction: {predicted_label}")
