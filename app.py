import streamlit as st
import requests
import PIL.Image    

# Two columns
col1, col2 = st.columns([2, 2])

with col1:
    st.write("""
            #### Welcome to our App
            This is a simple web app that uses a deep learning model to find similar faces in images.
            You can upload an image and find similar faces in the image.
            You can also take a picture using your camera and find similar faces in the image.
            """)
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/image_search.py", label="Search Image by Image", icon="1️⃣")
    st.page_link("pages/human_animals.py", label="Find animals with similar face", icon="2️⃣")
    st.page_link("pages/face_similarity.py", label="Face Similarity", icon="3️⃣")
    st.page_link("http://www.google.com", label="Google", icon="🌎")
with col2:  
    st.image(PIL.Image.open("front.png"), caption="Welcome to our App", use_column_width="auto")