import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Advanced Cancer Detection App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = r"C:\Users\asus\Documents\cancer_detection\models\cancer_detection_model.keras"
IMAGE_SIZE = (224, 224)

# Custom CSS to improve the app's appearance
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #34495e;
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    .stAlert {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_cache():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return load_model(MODEL_PATH)

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_cancer(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return float(prediction[0][0])

def main():
    st.title("🔬 Advanced Cancer Detection App")
    st.write("Upload a medical image to assess the probability of cancer.")

    try:
        model = load_model_cache()
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    try:
                        probability = predict_cancer(uploaded_file, model)
                        
                        with col2:
                            st.subheader("Prediction Results")
                            st.metric("Cancer Probability", f"{probability:.2%}")
                            
                            st.progress(probability)
                            
                            if probability > 0.7:
                                st.error("⚠️ High probability of cancer detected. Please consult a medical professional immediately.")
                            elif probability > 0.3:
                                st.warning("⚠️ Moderate probability of cancer. Further examination is recommended.")
                            else:
                                st.success("✅ Low probability of cancer. However, regular check-ups are always advisable.")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

    st.sidebar.title("About the App")
    st.sidebar.info(
        "This advanced cancer detection app utilizes a state-of-the-art deep learning model "
        "to analyze medical images and predict the probability of cancer. "
        "Our goal is to assist medical professionals in early detection and diagnosis."
    )

    st.sidebar.title("How It Works")
    st.sidebar.write(
        "1. Upload a medical image (e.g., X-ray, MRI, CT scan)\n"
        "2. Click 'Analyze Image'\n"
        "3. Review the prediction results and probability\n"
        "4. Consult with a healthcare professional for proper diagnosis"
    )

    st.sidebar.title("Disclaimer")
    st.sidebar.warning(
        "This application is for educational and demonstration purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider "
        "with any questions you may have regarding a medical condition."
    )

if __name__ == "__main__":
    main()