import streamlit as st
from PIL import Image

st.header("Face Similarity")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload First Image")
    image1 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image1")

with col2:
    st.subheader("Upload Second Image")
    image2 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image2")

if image1 and image2:
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption='First Image', use_column_width=True)
    with col2:
        st.image(img2, caption='Second Image', use_column_width=True)
    st.write("Similarity score: (to be implemented)")