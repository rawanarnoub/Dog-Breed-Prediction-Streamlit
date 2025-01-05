# Library imports
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf

# Function to load the model
def load_model_custom(model_path):
    try:
        # Attempt to load using TensorFlow's load_model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model_path = 'model.h5'  
model = load_model_custom(model_path)

if not model:
    st.error("Model loading failed. Please check the model file and try again.")
    st.stop()

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])  
submit = st.button('Predict')


if submit:
    if dog_image is not None:
        try:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
            
            # Display the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")
            
            # Preprocess the image for the model
            opencv_image = cv2.resize(opencv_image, (224, 224))  # Resize to model input size
            opencv_image = opencv_image / 255.0  # Normalize pixel values to [0, 1]
            opencv_image = np.expand_dims(opencv_image, axis=0)  # Add batch dimension
            
            # Predict the dog breed
            Y_pred = model.predict(opencv_image)
            predicted_class = CLASS_NAMES[np.argmax(Y_pred)]
            
            # Display the prediction
            st.title(f"The Dog Breed is: {predicted_class}")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.warning("Please upload an image before predicting.")
