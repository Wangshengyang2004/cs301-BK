import streamlit as st
from PIL import Image

def main():
    st.title('Image Processing Application')

    # Custom Navigation using st.page_link
    st.sidebar.title("Navigation")
    st.sidebar.page_link("Home", label="🏠 Home", icon="🏠")
    st.sidebar.page_link("face_similarity.py", label="👥 Face Similarity", icon="👥")
    st.sidebar.page_link("image_search.py", label="🔍 Image Search", icon="🔍")

    # Assume these are external links as examples
    st.sidebar.page_link("http://www.example.com", label="🌐 Visit Our Website", icon="🌐")


    st.header("Welcome to the Image Processing Application")
    st.write("This application provides two main functionalities:")
    st.write("1. **Face Similarity**: Compare two faces to determine how similar they are.")
    st.write("2. **General Image Search**: Upload an image to search for similar images.")

    

if __name__ == "__main__":
    main()
