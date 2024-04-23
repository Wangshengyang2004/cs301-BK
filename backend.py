from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import faiss
from backups.feature_extract import FeatureExtract

app = FastAPI()

# Load FAISS index and name list
index = faiss.read_index('VOCdevkit/voc.index')
name_list = np.load('VOCdevkit/name_list.npy')

class ImageData(BaseModel):
    image_base64: str

def decode_image(base64_str):
    """Decode base64 string to an image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = np.array(image.convert('RGB'))
    return image

def encode_image(image_path):
    """Convert an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/search/")
async def search_image(data: ImageData):
    try:
        image = decode_image(data.image_base64)
        feat_extractor = FeatureExtract()
        query_feature = feat_extractor.extractFeat(image)
        query_feature = np.expand_dims(query_feature, axis=0)
        D, I = index.search(query_feature, 5)  # Let's say we want top 5 results
        
        result_images = [encode_image(f'../VOCdevkit/VOC2012/JPEGImages/{name_list[i]}.jpg') for i in I[0]]
        return {"result_images": result_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
