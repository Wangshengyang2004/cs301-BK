import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# Function to convert image file to base64

def image_to_base64(image):
    buffered = BytesIO()
    # Convert RGBA to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Page title
st.title("Search Image by Image")

# Sidebar configuration
with st.sidebar:
    st.title("Select a Image type")
    photo_type = st.radio("", ("General Image", "Human Face Image"))

# Layout the page into two columns
col1, col2 = st.columns([2, 3])

# Middle column for uploading images
with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Please select the image to be uploaded", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Convert to base64
        encoded_image = image_to_base64(image)

        # Backend endpoint based on photo type
        if photo_type == "General Image":
            backend_endpoint = "http://127.0.0.1:8088/search/general"  # Update this URL to your backend
        elif photo_type == "Human Face Image":
            backend_endpoint =  "http://127.0.0.1:8088/search/human_face"  # Update if different from normal photos

# Right column for backend outputs
with col2:
    st.title("Output")
    # Backend request button
    if uploaded_file is not None and st.button("Process"):
        with st.spinner('Handling...'):
            # Send the request to the backend
            response = requests.post(backend_endpoint, json={"image_base64": encoded_image})
            if response.status_code == 200:
                outputs = [Image.open(BytesIO(base64.b64decode(img))) for img in response.json()['result_images']]
                for i, output in enumerate(outputs, start=1):
                    st.subheader(f"Output {i}")
                    st.image(output, caption=f"Matched Images {i}")
            else:
                # Error handling

                st.error("Failed to retrieve images from backend.")
