import numpy as np
import streamlit as st
from PIL import Image

# Function to simulate feature extraction
def extract_features(image):
    """Extract 512-dimensional features from the image using a pre-trained model."""
    return np.random.rand(512)

# Function to calculate cosine similarity
def calculate_similarity(features1, features2):
    """Calculate cosine similarity between two feature vectors."""
    dot_product = np.dot(features1, features2)
    norm_a = np.linalg.norm(features1)
    norm_b = np.linalg.norm(features2)
    return dot_product / (norm_a * norm_b)

st.header("Face Similarity")

# Sidebar options for input types for both images
with st.sidebar:
    st.subheader("Choose Input Type for Each Image")
    input_type1 = st.radio("Input for First Image", ["Upload", "Camera"], key="1")
    input_type2 = st.radio("Input for Second Image", ["Upload", "Camera"], key="2")

col1, col2 = st.columns(2)

# Handling first image input based on user selection
with col1:
    st.subheader("First Image Input")
    if input_type1 == "Upload":
        image1 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image1")
        if image1:
            img1 = Image.open(image1)
            st.image(img1, caption='First Image', use_column_width=True)
    elif input_type1 == "Camera":
        img_file_buffer1 = st.camera_input("Take a picture", key="image1")
        if img_file_buffer1:
            img1 = Image.open(img_file_buffer1)
            st.image(img1, caption='First Image', use_column_width=True)

# Handling second image input based on user selection
with col2:
    st.subheader("Second Image Input")
    if input_type2 == "Upload":
        image2 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image2")
        if image2:
            img2 = Image.open(image2)
            st.image(img2, caption='Second Image', use_column_width=True)
    elif input_type2 == "Camera":
        img_file_buffer2 = st.camera_input("Take a picture", key="image2")
        if img_file_buffer2:
            img2 = Image.open(img_file_buffer2)
            st.image(img2, caption='Second Image', use_column_width=True)

# Check if both images are available
if 'img1' in locals() and 'img2' in locals() and img1 and img2:
    if st.button("Calculate Similarity"):
        try:
            # Convert to RGB if necessary
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')

            # Extract features
            features1 = extract_features(img1)
            features2 = extract_features(img2)
            
            # Calculate similarity
            similarity_score = calculate_similarity(features1, features2)
            
            # Display the similarity score
            st.write(f"Similarity score: {similarity_score:.2f}")
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
else:
    st.warning("Please provide both images to calculate similarity.")
