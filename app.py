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
st.title("前端应用")

# Sidebar configuration
with st.sidebar:
    st.title("选择照片类型")
    photo_type = st.radio("", ("普通照片", "人像照片"))

# Layout the page into two columns
col1, col2 = st.columns([2, 3])

# Middle column for uploading images
with col1:
    st.subheader("上传照片")
    uploaded_file = st.file_uploader("请选择要上传的照片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的照片", use_column_width=True)
        # Convert to base64
        encoded_image = image_to_base64(image)

        # Backend endpoint based on photo type
        if photo_type == "普通照片":
            backend_endpoint = "http://192.168.31.120:8008/search/"  # Update this URL to your backend
        elif photo_type == "人像照片":
            backend_endpoint =  "http://192.168.31.120:8008/search/"  # Update if different from normal photos

# Right column for backend outputs
with col2:
    st.title("后端输出")
    # if 'outputs' in locals():
    #     for i, output in enumerate(outputs, start=1):
    #         st.subheader(f"输出 {i}")
    #         st.image(output, caption=f"匹配的图像 {i}")

    # Backend request button
    if uploaded_file is not None and st.button("获取输出"):
        with st.spinner('正在处理中...'):
            # Send the request to the backend
            response = requests.post(backend_endpoint, json={"image_base64": encoded_image})
            if response.status_code == 200:
                outputs = [Image.open(BytesIO(base64.b64decode(img))) for img in response.json()['result_images']]
                for i, output in enumerate(outputs, start=1):
                    st.subheader(f"输出 {i}")
                    st.image(output, caption=f"匹配的图像 {i}")
            else:
                st.error("Failed to retrieve images from backend.")
