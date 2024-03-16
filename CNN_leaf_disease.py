import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to load and predict with the model
def predict_disease(image):
    # Custom loading function to handle InputLayer issue
    def custom_object(name):
        if name == 'InputLayer':
            return tf.keras.layers.InputLayer(input_shape=(224, 224, 3), batch_size=None)
        else:
            return getattr(tf.keras.layers, name)

    # Load the trained model with custom objects
    try:
        model = tf.keras.models.load_model('model.h5', custom_objects=custom_object)
    except Exception as e:
        return str(e)

    # Preprocess the image
    image = preprocess_image(image)

    # Make prediction
    try:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        return predicted_class
    except Exception as e:
        return str(e)

# Streamlit app title
st.title("Leaf Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the disease
        prediction_class = predict_disease(image)

        # Class mapping
        class_mapping = {
            0: "Healthy",
            1: "Powdery Mildew",
            2: "Rust"
        }

        prediction_label = class_mapping.get(prediction_class, "Unknown")
        st.success(f"Prediction: {prediction_label}")
    except Exception as e:
        st.error(f"Error: {e}")
