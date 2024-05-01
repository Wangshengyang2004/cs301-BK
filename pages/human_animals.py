# Input: Human Face Image, or Camera click
# Output: Top_N similar animal images
import requests
from PIL import Image
import base64
from io import BytesIO
import streamlit as st
import numpy as np

def image_to_base64(image):
    buffered = BytesIO()
    # Convert RGBA to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Page title
st.title("Human Face to Animal Face")
with st.sidebar:
    st.title("Select a Input type")
    photo_type = st.radio("", ("Upload Image", "Camera"))

# Layout the page into two columns
col1, col2 = st.columns([2, 3])

if photo_type == "Upload Image":
    # Middle column for uploading images
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Please select the image to be uploaded", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Convert to base64
            encoded_image = image_to_base64(image)
            backend_endpoint = "http://127.0.0.1:8088/search/human_face"

elif photo_type == "Camera":
    with col1:
        st.subheader("Take a picture using the WebCam")
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer:
            st.image(img_file_buffer)
            if img_file_buffer is not None:
                # To read image file buffer as a PIL Image:
                img = Image.open(img_file_buffer)
                # To convert PIL Image to numpy array:
                img_array = np.array(img)
                encoded_image = image_to_base64(img)
                backend_endpoint = "http://127.0.0.1:8088/search/human_face"
        else:
            st.error("No picture taken, please check your camera")


with col2:
    st.title("Output")
    # Backend request button
    if st.button("Process"):
        with st.spinner('Handling...'):
            # Send the request to the backend
            response = requests.post(backend_endpoint, json={"image_base64": encoded_image})
            if response.status_code == 200:
                outputs = [Image.open(BytesIO(base64.b64decode(img))) for img in response.json()['result_images']]
                if len(outputs) == 0:
                    st.error("No similar animal images found. Please try another image. \n Possible reasons: \n 1. The image does not contain a human face. \n 2. The human face in the image is not clear. \n 3. The human face in the image is not facing the camera. \n 4. The human face in the image is not detected.")
                for i, output in enumerate(outputs, start=1):
                    st.subheader(f"Output {i}")
                    st.image(output, caption=f"Matched Images {i}")
            else:
                st.error("Failed to retrieve images from backend.")