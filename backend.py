from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import faiss
import requests
from io import BytesIO
from PIL import Image
import base64
from backups.feature_extract import FeatureExtract
app = FastAPI()
face_detector = cv2.dnn.readNetFromCaffe('/home/simonwsy/workspace/cs301-BK/images/faces/weights/deploy.prototxt.txt'
                                         ,'/home/simonwsy/workspace/cs301-BK/images/faces/weights/res10_300x300_ssd_iter_140000.caffemodel')



# Load FAISS index and name list
general_index = faiss.read_index('VOCdevkit/voc.index')
general_name_list = np.load('VOCdevkit/name_list.npy')
face_index = faiss.read_index('afhq/afhq.index')
face_name_list = np.load('afhq/name_list.npy')

class ImageData(BaseModel):
    image_base64: str

def decode_image(base64_str):
    """Decode base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def encode_image(image):
    """Encode a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_embedding(image_bytes):
    """Send image to prediction model and get embedding."""
    url = "http://127.0.0.1:8080/predictions/facenet"  # Modify as necessary to the correct URL
    headers = {'Content-Type': 'image/jpeg'}
    response = requests.post(url, headers=headers, data=image_bytes.getvalue())
    if response.status_code == 200:
        return np.array(response.json()['res'])  # Adjust this depending on the actual response format
    else:
        raise ValueError("Failed to get embedding from model: " + response.text)

def get_cropped_face(image):
    """Detect and crop the largest face found in the image."""
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (300, 300))
    blob = cv2.dnn.blobFromImage(img_resized, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_detector.setInput(blob)
    detections = face_detector.forward()
    if len(detections) > 0 and detections[0, 0, 0, 2] > 0.15:
        x1, y1, x2, y2 = (detections[0, 0, 0, 3:7] * np.array([image.width, image.height, image.width, image.height])).astype(int)
        return Image.fromarray(img_np[y1:y2, x1:x2])
    return None

@app.post("/search/general")
async def search_image(data: ImageData):
    try:
        image = decode_image(data.image_base64)
        image_io = BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        # Assuming `extract_feature` is defined elsewhere and can take a BytesIO
        extract_feature = FeatureExtract().extractFeat
        feature = extract_feature(image_io)
        D, I = general_index.search(feature[np.newaxis], 5)
        result_images = [encode_image(Image.open(f'./VOCdevkit/VOC2012/JPEGImages/{general_name_list[i]}.jpg')) for i in I[0]]
        return {"result_images": result_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/human_face")
async def search_face_image(data: ImageData):
    try:
        image = decode_image(data.image_base64)
        cropped_face = get_cropped_face(image)
        if cropped_face is not None:
            cropped_face_io = BytesIO()
            cropped_face.save(cropped_face_io, format='JPEG')
            cropped_face_io.seek(0)
            # embedding = get_embedding(cropped_face_io)  # Assuming this function now takes BytesIO
            feature = FeatureExtract().extractFeat(cropped_face_io)
            D, I = face_index.search(feature[np.newaxis], 5)
            result_images = [encode_image(Image.open(f'./afhq/train/{face_name_list[i]}.jpg')) for i in I[0]]
            return {"result_images": result_images}
        else:
            return {"result_images": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)